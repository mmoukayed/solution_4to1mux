#!/usr/bin/env python3
"""Make an MP4 from record.py's frames: the arena camera full frame, the head camera inset.
Frames are repeated by their simulation time stamps, so the video plays at simulation speed
(1 s of video = 1 s of robot time) however slowly the simulator ran.

    python3 make_video.py REC_DIR OUT.mp4 [--label TEXT] [--fps 15]
"""
import argparse
import os
import subprocess

import cv2
import numpy as np


def stamps(d):
    return [(int(k), float(t)) for k, t in (ln.split() for ln in open(os.path.join(d, 'stamps.txt')) if ln.strip())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rec')
    ap.add_argument('out')
    ap.add_argument('--label', default='')
    ap.add_argument('--fps', type=float, default=15)
    a = ap.parse_args()
    main_dir, inset_dir = os.path.join(a.rec, 'arena'), os.path.join(a.rec, 'head')
    M, I = stamps(main_dir), stamps(inset_dir)
    if not M:
        raise SystemExit('no arena frames in %s' % main_dir)
    it = np.array([t for _, t in I]) if I else None
    W, H = 1280, 720
    raw = a.out + '.raw.mp4'
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*'mp4v'), a.fps, (W, H))
    t0, written = M[0][1], 0
    for idx, (k, t) in enumerate(M):
        f = cv2.imread(os.path.join(main_dir, 'f%05d.jpg' % k))
        if f is None:
            continue
        f = cv2.resize(f, (W, H))
        if it is not None:
            g = cv2.imread(os.path.join(inset_dir, 'f%05d.jpg' % I[int(np.argmin(np.abs(it - t)))][0]))
            if g is not None:
                iw, ih = 448, 252
                x0, y0 = W - iw - 16, H - ih - 16
                cv2.rectangle(f, (x0 - 3, y0 - 3), (x0 + iw + 2, y0 + ih + 2), (255, 255, 255), -1)
                f[y0:y0 + ih, x0:x0 + iw] = cv2.resize(g, (iw, ih))
        text = '%s | t %.1f s' % (a.label, t - t0) if a.label else 't %.1f s' % (t - t0)
        cv2.putText(f, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        cv2.putText(f, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 1)
        t_next = M[idx + 1][1] if idx + 1 < len(M) else t + 1.0 / a.fps
        repeat = max(1, int(round((t_next - t0) * a.fps)) - written)
        for _ in range(repeat):
            vw.write(f)
        written += repeat
    vw.release()
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', raw, '-c:v', 'libx264', '-crf', '23',
                    '-pix_fmt', 'yuv420p', a.out], check=False)
    if os.path.exists(a.out):
        os.remove(raw)
    print('wrote %s (%.1f s at simulation speed)' % (a.out, written / a.fps))


if __name__ == '__main__':
    main()
