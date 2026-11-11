// A RealSense D435's depth, made from the renderer's exact depth.
//
// The numbers are measured on the real robot's head camera (slot 1,
// 2026-10-31: a white three-bay shelf in a library wall, square on from
// 2.65 m, 848 x 480):
//   - errors come in patches, not pixels: neighbouring pixels agree (0.98 at
//     1 px, 0.56 at 8 px, 0 at 16 px), 60-90 mm about a plane at 2.5-2.7 m,
//     so sigma = k z^2 with k ~ 0.010 /m (the textbook D435 figure is 0.0036);
//   - the depth is smoothed over the stereo matcher's window, ~9 px: a shelf
//     board's 20 mm front at 2.7 m is ~3 px and is blurred away, so no depth
//     step marks a board;
//   - no return comes in specks AND big blobs (median 3-5 px, the largest
//     1100-3400 px; 87-96 % of the lost area is in blobs over 500 px), and
//     it gathers at depth STEPS - board and divider edges, where one imager
//     sees what the other cannot: 47-67 % of the pixels near a step read
//     nothing against 2-27 % elsewhere; and the left edge of the image is the
//     stereo shadow, baseline x focal / depth pixels wide (61-99 % empty in
//     the first 20 columns at 848). An even scatter of small holes, the
//     first model, looked nothing like it (2026-11-01, measured on three
//     slot-1 frames: src/tools in the session notes).
//
// Each frame takes its noise from fixed correlated fields at random offsets:
// the noise is fresh every frame, the cost a few lookups a pixel.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace mujoco_ros2_control_plugins
{

struct DepthModelParams
{
  double noise_k{ 0.010 };       // sigma_z = noise_k * z^2, metres
  double pixel_share{ 0.15 };    // of sigma, uncorrelated pixel to pixel
  int noise_patch_px{ 5 };       // box radius the patch noise is smoothed over
  int blur_px{ 4 };              // box radius of the matcher's smoothing (9 px window)
                                 // (every px here at 848 px wide; scaled to the image)
  double holes_dark{ 0.08 };     // share of a dark surface lost in big blobs, away from any step
  double holes_bright{ 0.14 };   // ...and of bright white
  int hole_patch_px{ 20 };       // box radius of those blobs
  double edge_holes{ 0.60 };     // share of the band round a depth step that reads nothing
  int edge_band_px{ 8 };         // that band's half-width
  int edge_patch_px{ 6 };        // box radius of the edge blobs (long along the edge by the band)
  double step_m{ 0.15 };         // a depth step: this much across a few px, after the blur (a board against its bay; not book spines)
  double specks{ 0.02 };         // share lost in specks a few pixels across
  double big_holes{ 0.02 };      // share of the image lost in very large blobs
  int big_hole_patch_px{ 40 };
  double baseline_m{ 0.050 };    // the D435's stereo baseline: the left shadow band
  double fx{ 0.0 };              // the image's focal length in px (configure); 0: no shadow band
  uint32_t seed{ 7 };
};

class DepthModel
{
public:
  void configure(const DepthModelParams& p, uint32_t width, uint32_t height)
  {
    p_ = p;
    // the pixel sizes are a D435's at 848 px wide: the same angle on this image
    const double f = width / 848.0;
    auto px = [f](int r) { return r > 0 ? std::max(1, static_cast<int>(std::lround(r * f))) : 0; };
    p_.noise_patch_px = px(p.noise_patch_px);
    p_.blur_px = px(p.blur_px);
    p_.hole_patch_px = px(p.hole_patch_px);
    p_.big_hole_patch_px = px(p.big_hole_patch_px);
    p_.edge_band_px = px(p.edge_band_px);
    p_.edge_patch_px = px(p.edge_patch_px);
    w_ = width;
    h_ = height;
    fw_ = 2 * w_;
    fh_ = 2 * h_;
    rng_.seed(p_.seed);
    noise_ = field(p_.noise_patch_px);
    white_ = field(0);
    holes_ = field(p_.hole_patch_px);
    big_ = field(p_.big_hole_patch_px);
    edge_ = field(p_.edge_patch_px);
    speck_ = field(std::max(1, static_cast<int>(std::lround(1.5 * f))));
    // thresholds by quantile of the (unit, roughly normal) hole field
    // sampled across the WHOLE field (a stride, not the first rows): the
    // big-hole field is smooth over 25 px and its first rows are only a few
    // of its blobs
    std::vector<float> s = sample(holes_);
    std::sort(s.begin(), s.end());
    hole_q_.resize(101);
    for (int i = 0; i <= 100; ++i)
    {
      hole_q_[i] = s[std::min(s.size() - 1, static_cast<size_t>((1.0 - i / 100.0) * static_cast<double>(s.size() - 1)))];
    }
    std::vector<float> b = sample(big_);
    std::sort(b.begin(), b.end());
    big_t_ = b[std::min(b.size() - 1, static_cast<size_t>((1.0 - p_.big_holes) * static_cast<double>(b.size() - 1)))];
    edge_t_ = quantile(edge_, p_.edge_holes);
    speck_t_ = quantile(speck_, p_.specks);
    band_.assign(static_cast<size_t>(w_) * h_, 0.0f);
    band_tmp_.assign(band_.size(), 0.0f);
    blurred_.assign(static_cast<size_t>(w_) * h_, 0.0f);
    sum_.assign(blurred_.size(), 0.0f);
    cnt_.assign(blurred_.size(), 0.0f);
    ready_ = true;
  }

  bool ready() const
  {
    return ready_;
  }

  // the parameters as applied (pixel sizes scaled to this image)
  const DepthModelParams& applied() const
  {
    return p_;
  }

  // depth: w x h metres, 0 = no return (in place). rgb: w x h x 3, same rows.
  void apply(float* depth, const uint8_t* rgb)
  {
    blur(depth);
    steps(blurred_.data());   // the steps the matcher still sees: a spine's 2-3 cm against
                              // its neighbour is gone in its window; a board against its bay is not
    std::uniform_int_distribution<uint32_t> ox(0, fw_ - w_), oy(0, fh_ - h_);
    const uint32_t nx = ox(rng_), ny = oy(rng_), wx = ox(rng_), wy = oy(rng_);
    const uint32_t hx = ox(rng_), hy = oy(rng_), bx = ox(rng_), by = oy(rng_);
    const uint32_t ex = ox(rng_), ey = oy(rng_), sx = ox(rng_), sy = oy(rng_);
    // how bright each place is, smoothed over ~15 px: per pixel, the holes'
    // edges followed every spine's colour and came out as vertical streaks
    for (size_t i = 0; i < static_cast<size_t>(w_) * h_; ++i)
    {
      const uint8_t* c = rgb + 3 * i;
      band_tmp_[i] = static_cast<float>((c[0] + c[1] + c[2]) / (3.0 * 255.0));
    }
    {
      const int r = std::max(1, static_cast<int>(std::lround(7.0 * w_ / 848.0)));
      lum_.resize(band_tmp_.size());
      std::vector<float> tmp(band_tmp_.size());
      box(band_tmp_.data(), tmp.data(), w_, h_, r, 1, w_);
      box(tmp.data(), lum_.data(), h_, w_, r, w_, 1);
    }
    // the stereo shadow's width, once a frame, from the median depth of the
    // image's left tenth (per pixel, the noise frayed it into stripes)
    uint32_t shadow = 0;
    if (p_.fx > 0.0)
    {
      std::vector<float> left;
      left.reserve(static_cast<size_t>(h_) * (w_ / 10 + 1));
      for (uint32_t y = 0; y < h_; ++y)
      {
        for (uint32_t x = 0; x < w_ / 10; ++x)
        {
          const float z = blurred_[static_cast<size_t>(y) * w_ + x];
          if (z > 0.0f)
          {
            left.push_back(z);
          }
        }
      }
      if (!left.empty())
      {
        std::nth_element(left.begin(), left.begin() + static_cast<long>(left.size() / 2), left.end());
        shadow = static_cast<uint32_t>(p_.fx * p_.baseline_m / static_cast<double>(left[left.size() / 2]));
      }
    }
    const float patch = static_cast<float>(std::sqrt(1.0 - p_.pixel_share * p_.pixel_share));
    const float pix = static_cast<float>(p_.pixel_share);
    for (uint32_t y = 0; y < h_; ++y)
    {
      for (uint32_t x = 0; x < w_; ++x)
      {
        const size_t i = static_cast<size_t>(y) * w_ + x;
        float z = blurred_[i];
        if (z <= 0.0f)
        {
          depth[i] = 0.0f;
          continue;
        }
        // no return. The stereo shadow at the left edge: what the left
        // imager sees past the right one's field, baseline x focal / z wide
        if (x < shadow)
        {
          depth[i] = 0.0f;
          continue;
        }
        // big blobs round a depth step, where one imager sees what the
        // other cannot; big blobs elsewhere, more on bright surfaces; specks
        const double t = std::clamp((static_cast<double>(lum_[i]) - 0.55) / 0.40, 0.0, 1.0);
        const double share = p_.holes_dark + (p_.holes_bright - p_.holes_dark) * t;
        const int q = static_cast<int>(std::lround(std::clamp(share, 0.0, 1.0) * 100.0));
        const bool near_step = band_[i] > 0.0f && z > near_[i] + 0.5f * static_cast<float>(p_.step_m);
        if ((near_step && at(edge_, ex + x, ey + y) > edge_t_) || at(holes_, hx + x, hy + y) > hole_q_[q] ||
            at(big_, bx + x, by + y) > big_t_ || at(speck_, sx + x, sy + y) > speck_t_)
        {
          depth[i] = 0.0f;
          continue;
        }
        const float n = patch * at(noise_, nx + x, ny + y) + pix * at(white_, wx + x, wy + y);
        depth[i] = std::max(0.0f, z + n * static_cast<float>(p_.noise_k) * z * z);
      }
    }
  }

private:
  static std::vector<float> sample(const std::vector<float>& f)
  {
    const size_t stride = std::max<size_t>(1, f.size() / 200000);
    std::vector<float> out;
    out.reserve(f.size() / stride + 1);
    for (size_t i = 0; i < f.size(); i += stride)
    {
      out.push_back(f[i]);
    }
    return out;
  }

  float quantile(const std::vector<float>& f, double share) const
  {
    std::vector<float> s = sample(f);
    std::sort(s.begin(), s.end());
    const double k = std::clamp(1.0 - share, 0.0, 1.0) * static_cast<double>(s.size() - 1);
    return s[std::min(s.size() - 1, static_cast<size_t>(k))];
  }

  // band_: > 0 within edge_band_px of a depth step of `depth` (the blurred one)
  void steps(const float* depth)
  {
    const float step = static_cast<float>(p_.step_m);
    // across a few pixels: the blur has spread every step over its window
    const uint32_t d = static_cast<uint32_t>(std::max(1, static_cast<int>(std::lround(2.0 * w_ / 848.0))));
    for (uint32_t y = 0; y < h_; ++y)
    {
      for (uint32_t x = 0; x < w_; ++x)
      {
        const size_t i = static_cast<size_t>(y) * w_ + x;
        float m = 0.0f;
        if (x >= d && x + d < w_)
        {
          const float a = depth[i - d], b = depth[i + d];
          if (a > 0.0f && b > 0.0f && std::fabs(a - b) > step)
          {
            m = 1.0f;
          }
        }
        if (y >= d && y + d < h_)
        {
          const float a = depth[i - d * w_], b = depth[i + d * w_];
          if (a > 0.0f && b > 0.0f && std::fabs(a - b) > step)
          {
            m = 1.0f;
          }
        }
        band_tmp_[i] = m;
      }
    }
    const int r = std::max(1, p_.edge_band_px);
    box(band_tmp_.data(), band_.data(), w_, h_, r, 1, w_);
    box(band_.data(), band_tmp_.data(), h_, w_, r, w_, 1);
    std::copy(band_tmp_.begin(), band_tmp_.end(), band_.begin());
    // ...and the nearest depth within the same radius: only what lies BEHIND
    // a nearer edge is lost - one imager sees it, the other's view of it is
    // blocked by the edge - while the near surface keeps its depth (a tray's
    // rim, a board's front). Holes on both sides lost every tray rim (4 of
    // 10 runs could not find the tray, 2026-11-01).
    near_.resize(band_.size());
    min_filter(depth, band_tmp_.data(), w_, h_, r, 1, w_);
    min_filter(band_tmp_.data(), near_.data(), h_, w_, r, w_, 1);
  }

  // a running minimum of radius r over n lines (0 = no return, ignored)
  static void min_filter(const float* in, float* out, uint32_t len, uint32_t n, int r, size_t step, size_t line)
  {
    const float inf = 1e9f;
    for (uint32_t l = 0; l < n; ++l)
    {
      const float* a = in + l * line;
      float* b = out + l * line;
      for (int j = 0; j < static_cast<int>(len); ++j)
      {
        float m = inf;
        for (int k = std::max(0, j - r); k <= std::min(static_cast<int>(len) - 1, j + r); ++k)
        {
          const float v = a[static_cast<size_t>(k) * step];
          if (v > 0.0f && v < m)
          {
            m = v;
          }
        }
        b[static_cast<size_t>(j) * step] = m >= inf ? 0.0f : m;
      }
    }
  }

  float at(const std::vector<float>& f, uint32_t x, uint32_t y) const
  {
    return f[static_cast<size_t>(y) * fw_ + x];
  }

  // unit-variance noise smoothed by three box passes of radius r (~gaussian)
  std::vector<float> field(int r)
  {
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> f(static_cast<size_t>(fw_) * fh_);
    for (auto& v : f)
    {
      v = g(rng_);
    }
    if (r > 0)
    {
      std::vector<float> tmp(f.size());
      for (int pass = 0; pass < 3; ++pass)
      {
        box(f.data(), tmp.data(), fw_, fh_, r, 1, fw_);   // along rows
        box(tmp.data(), f.data(), fh_, fw_, r, fw_, 1);   // along columns
      }
    }
    double m = 0, s = 0;
    for (float v : f)
    {
      m += v;
    }
    m /= static_cast<double>(f.size());
    for (float v : f)
    {
      s += (v - m) * (v - m);
    }
    const float sd = static_cast<float>(std::sqrt(s / static_cast<double>(f.size())));
    for (auto& v : f)
    {
      v = static_cast<float>((v - m) / sd);
    }
    return f;
  }

  // a running-sum box filter of radius r over n lines of len samples; one line
  // starts `line` apart, samples within it `step` apart (edges clamped)
  static void box(const float* in, float* out, uint32_t len, uint32_t n, int r, size_t step, size_t line)
  {
    const float k = 1.0f / static_cast<float>(2 * r + 1);
    for (uint32_t l = 0; l < n; ++l)
    {
      const float* a = in + l * line;
      float* b = out + l * line;
      auto s = [&](int j) { return a[static_cast<size_t>(std::clamp(j, 0, static_cast<int>(len) - 1)) * step]; };
      float acc = 0;
      for (int j = -r; j <= r; ++j)
      {
        acc += s(j);
      }
      for (int j = 0; j < static_cast<int>(len); ++j)
      {
        b[static_cast<size_t>(j) * step] = acc * k;
        acc += s(j + r + 1) - s(j - r);
      }
    }
  }

  // the matcher's window: a mean over the valid depths within blur_px
  void blur(const float* depth)
  {
    const int r = p_.blur_px;
    const size_t n = static_cast<size_t>(w_) * h_;
    if (r <= 0)
    {
      std::copy(depth, depth + n, blurred_.begin());
      return;
    }
    for (uint32_t y = 0; y < h_; ++y)
    {
      for (uint32_t x = 0; x < w_; ++x)
      {
        const size_t i = static_cast<size_t>(y) * w_ + x;
        cnt_[i] = depth[i] > 0.0f ? 1.0f : 0.0f;
        sum_[i] = depth[i] > 0.0f ? depth[i] : 0.0f;
      }
    }
    std::vector<float>& ts = tmp_s_;
    std::vector<float>& tc = tmp_c_;
    ts.resize(n);
    tc.resize(n);
    box(sum_.data(), ts.data(), w_, h_, r, 1, w_);
    box(cnt_.data(), tc.data(), w_, h_, r, 1, w_);
    box(ts.data(), sum_.data(), h_, w_, r, w_, 1);
    box(tc.data(), cnt_.data(), h_, w_, r, w_, 1);
    for (size_t i = 0; i < n; ++i)
    {
      // a pixel with no return of its own stays without one
      blurred_[i] = (depth[i] > 0.0f && cnt_[i] > 1e-6f) ? sum_[i] / cnt_[i] : 0.0f;
    }
  }

  DepthModelParams p_;
  uint32_t w_{ 0 }, h_{ 0 }, fw_{ 0 }, fh_{ 0 };
  std::mt19937 rng_;
  std::vector<float> noise_, white_, holes_, big_, edge_, speck_;
  std::vector<float> hole_q_;
  float big_t_{ 0 }, edge_t_{ 0 }, speck_t_{ 0 };
  std::vector<float> band_, band_tmp_, lum_, near_;
  std::vector<float> blurred_, sum_, cnt_, tmp_s_, tmp_c_;
  bool ready_{ false };
};

}  // namespace mujoco_ros2_control_plugins
