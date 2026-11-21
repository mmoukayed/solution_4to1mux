"""Depth-supported book surfaces and conservative shelf-divider envelopes."""
import cv2
import numpy as np


def cloud(depth, transform, info, image_shape, origin=(0, 0)):
    height, width = image_shape
    dh, dw = depth.shape
    fx, fy = info.k[0]*dw/(info.width or width), info.k[4]*dh/(info.height or height)
    if fx <= 0 or fy <= 0:
        raise ValueError('Invalid camera intrinsics')
    cx, cy = info.k[2]*dw/(info.width or width), info.k[5]*dh/(info.height or height)
    if not np.isfinite([fx, fy, cx, cy]).all() or not np.isfinite(transform).all():
        raise ValueError('Nonfinite camera calibration')
    v, u = np.indices(depth.shape)
    u, v = u+origin[0], v+origin[1]
    points = np.stack([(u-cx)*depth/fx, (v-cy)*depth/fy, depth], axis=-1)
    return points @ transform[:3, :3].T+transform[:3, 3]


def measure_book(image, depth, transform, info, box, normal, white=False):
    """Grow a front-depth component from the OCR label; never segment white alone.

    Shelf boards/background merged with the seed are rejected by metric size.
    RGB and registered depth must have identical pixels, as in the live stack.
    """
    if image.shape[:2] != depth.shape:
        return None
    x, y, w, h = map(int, box)
    height, width = depth.shape
    if w < 2 or h < 2 or x < 0 or y < 0 or x+w > width or y+h > height:
        return None
    normal = np.asarray(normal, float)
    if len(normal) == 2:
        normal = np.r_[normal, 0.]
    tangent = np.array([-normal[1], normal[0], 0.])
    pad_x, pad_y = (max(18, 4*w), max(20, 2*h)) if white else (0, 0)
    x0, y0 = max(0, x-pad_x), max(0, y-pad_y)
    x1, y1 = min(width, x+w+pad_x), min(height, y+h+pad_y)
    d = depth[y0:y1, x0:x1]
    # Intrinsics use the original registered image dimensions.
    full_info = type('Info', (), {'k': list(info.k), 'width': width, 'height': height})()
    full_info.k[0] *= width/(info.width or width)
    full_info.k[4] *= height/(info.height or height)
    full_info.k[2] *= width/(info.width or width)
    full_info.k[5] *= height/(info.height or height)
    # cloud() scales its intrinsics by the patch size; retain full focal lengths.
    local_info = type('Info', (), {'k': full_info.k, 'width': d.shape[1], 'height': d.shape[0]})()
    try:
        points = cloud(d, transform, local_info, d.shape, (x0, y0))
    except ValueError:
        return None
    valid = np.isfinite(d) & (d > .20) & (d < 6.)
    seed = np.zeros(d.shape, bool)
    seed[y-y0:y-y0+h, x-x0:x-x0+w] = True
    seeds = points[seed & valid]
    if len(seeds) < 12:
        return None
    depths = seeds @ normal
    # The densest plane, not a few nearest noise pixels, anchors the component.
    counts, edges = np.histogram(depths, bins=max(1, int(np.ptp(depths)/.015)+1))
    i = int(np.argmax(counts))
    band = depths[(depths >= edges[i]) & (depths <= edges[i+1])]
    plane = float(np.median(band))
    mask = valid & (np.abs(points @ normal-plane) <= .018)
    if white:
        hsv = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        mask &= (hsv[..., 1] < 100) & ((hsv[..., 2] > 90) | seed)
        joined = cv2.morphologyEx(np.uint8(mask), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        # A standing book can touch the horizontal board at its bottom.
        # Find its lateral run at label height before growing vertically, so
        # that board cannot merge the book into the entire white shelf.
        band0, band1 = max(0, y-y0+h//4), min(d.shape[0], y-y0+3*h//4+1)
        profile = np.mean(joined[band0:band1], axis=0) > .25
        centre_x = int(np.clip(x-x0+w//2, 0, len(profile)-1))
        near = np.flatnonzero(profile & (np.abs(np.arange(len(profile))-centre_x) <= max(2, w//2)))
        if not len(near):
            return None
        centre_x = int(near[np.argmin(np.abs(near-centre_x))])
        left, right = centre_x, centre_x
        while left > 0 and profile[left-1]:
            left -= 1
        while right+1 < len(profile) and profile[right+1]:
            right += 1
        # A shelf board extends much farther sideways than the spine. Remove
        # those broad rows before computing the book's top/bottom and centre.
        broad_rows = np.count_nonzero(joined, axis=1) > 2.5*(right-left+1)
        joined[broad_rows] = 0
        joined[:, :left] = 0
        joined[:, right+1:] = 0
        mask &= joined > 0
        _, labels = cv2.connectedComponents(joined)
        choices, support = np.unique(labels[seed & mask], return_counts=True)
        nonzero = choices != 0
        if not np.any(nonzero):
            return None
        label = choices[nonzero][np.argmax(support[nonzero])]
        mask &= labels == label
    selected = points[mask]
    if len(selected) < 24:
        return None
    lateral, vertical = selected @ tangent, selected[:, 2]
    lo, hi = np.percentile(lateral, [2, 98])
    bottom, top = np.percentile(vertical, [2, 98])
    metric_width, metric_height = float(hi-lo), float(top-bottom)
    if not (.012 <= metric_width <= (.09 if white else .20) and .10 <= metric_height <= .40):
        return None
    centre = normal*float(np.median(selected @ normal))+tangent*(lo+hi)/2
    centre[2] = (bottom+top)/2
    ys, xs = np.nonzero(mask)
    return {'point': centre.tolist(), 'width': metric_width, 'height': metric_height,
            'box': [int(xs.min()+x0), int(ys.min()+y0), int(np.ptp(xs)+1), int(np.ptp(ys)+1)],
            'depth_supported': True}


def divider_bounds(depth, transform, info, image_shape, shelf, normal, height, half_width=.43):
    """Find side-facing depth clusters near the calibrated divider positions."""
    # Downsample with focal lengths scaled consistently.
    step = 4
    d = depth[::step, ::step]
    try:
        points = cloud(d, transform, info, image_shape).reshape(-1, 3)
    except ValueError:
        return -half_width+.03, half_width-.03, 'calibrated_with_margin'
    good = np.isfinite(points).all(axis=1) & np.isfinite(d.ravel()) & (d.ravel() > .20)
    points = points[good]
    n = np.asarray(normal)
    t = np.array([-n[1], n[0], 0.])
    offset = points-np.asarray(shelf)
    forward, across = offset @ n, offset @ t
    usable = (np.abs(points[:, 2]-height) < .12) & (forward > -.04) & (forward < .45)
    bounds, measured = [], []
    for expected in (-half_width, half_width):
        keep = usable & (np.abs(across-expected) < .09)
        values = across[keep]
        if len(values) >= 12:
            hist, bins = np.histogram(values, bins=np.arange(expected-.09, expected+.101, .015))
            peak = int(np.argmax(hist))
            selected = keep & (across >= bins[peak]) & (across <= bins[peak+1])
            if selected.sum() >= 12 and np.ptp(forward[selected]) > .06:
                bounds.append(float(np.median(across[selected])))
                measured.append(True)
                continue
        # Unobserved walls are moved inward, never optimistically widened.
        bounds.append(expected+(.03 if expected < 0 else -.03))
        measured.append(False)
    return bounds[0], bounds[1], 'depth' if all(measured) else 'calibrated_with_margin'


def validate_geometry(g):
    try:
        numbers = [*g['point'], *g['shelf_point'], *g['normal'], g['left'], g['right'], g['width'], g['height'], g['stamp']]
        return (all(len(g[name]) == 3 for name in ('point', 'shelf_point', 'normal')) and g['depth_supported'] is True and np.isfinite(numbers).all() and
                .012 <= g['width'] <= .20 and .10 <= g['height'] <= .40 and
                .45 <= g['right']-g['left'] <= 1.2 and
                abs(np.linalg.norm(g['normal'])-1) < .01 and abs(g['normal'][2]) < .01)
    except (KeyError, TypeError, ValueError):
        return False


def check_divider_points(points, radii, g, margin=.015):
    n, origin = np.asarray(g['normal']), np.asarray(g['shelf_point'])
    t = np.array([-n[1], n[0], 0.])
    for p, radius in zip(points, radii):
        relative = np.asarray(p)-origin
        forward, across = relative @ n, relative @ t
        if -.025-radius <= forward <= .50+radius:
            if min(abs(across-g['left']), abs(across-g['right'])) < radius+margin:
                raise ValueError('Divider clearance refused: arm/gripper envelope crosses a side board.')


def grasp_fits(g, half_width=.065, margin=.015):
    if not validate_geometry(g):
        return False
    normal = np.asarray(g['normal'])
    tangent = np.array([-normal[1], normal[0], 0.])
    across = (np.asarray(g['point'])-np.asarray(g['shelf_point'])) @ tangent
    envelope = max(half_width, g['width']/2+.015)+margin
    return bool(g['left']+envelope <= across <= g['right']-envelope)
