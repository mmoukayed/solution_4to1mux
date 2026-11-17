"""Surface-supported RGB-D measurements for hollow bins and narrow spines."""
import cv2
import numpy as np


def masked_surface(box, mask, depth, camera_to_world, info, maximum_depth=8.0):
    """Use a supported depth cluster on coloured pixels, never the hollow centre.

    Return the bounding-box centre at surface range and its apparent metric
    width/height. Unsupported, mixed or sparse depth is deliberately refused.
    """
    height, width = mask.shape
    dh, dw = depth.shape
    x, y, w, h = box
    x0, y0 = max(0, int(x*dw/width)), max(0, int(y*dh/height))
    x1, y1 = min(dw, int(np.ceil((x+w)*dw/width))), min(dh, int(np.ceil((y+h)*dh/height)))
    if x1 <= x0 or y1 <= y0:
        return None
    selected = cv2.resize(mask[y:y+h, x:x+w], (x1-x0, y1-y0), interpolation=cv2.INTER_NEAREST) > 0
    coloured = int(selected.sum())
    if coloured < 12:
        return None
    patch = depth[y0:y1, x0:x1]
    valid = selected & np.isfinite(patch) & (patch > .20) & (patch <= maximum_depth)
    values = patch[valid]
    if values.size < max(12, .20*coloured):
        return None
    # Mode rather than an unrestricted median: a ring may border several
    # background surfaces, or registered depth may have isolated foreground hits.
    counts, edges = np.histogram(values, bins=np.arange(.20, maximum_depth+.061, .06))
    peak = int(np.argmax(counts))
    seed = values[(values >= edges[peak]) & (values <= edges[peak+1])]
    if not seed.size:
        return None
    z = float(np.median(seed))
    cluster = values[np.abs(values-z) <= max(.06, .025*z)]
    if cluster.size < max(12, .45*values.size):
        return None
    z = float(np.median(cluster))
    fx = info.k[0]*dw/(info.width or width)
    fy = info.k[4]*dh/(info.height or height)
    cx = info.k[2]*dw/(info.width or width)
    cy = info.k[5]*dh/(info.height or height)
    if fx <= 0 or fy <= 0:
        return None
    u, v = (x+w/2)*dw/width, (y+h/2)*dh/height
    point = (camera_to_world @ np.array([(u-cx)*z/fx, (v-cy)*z/fy, z, 1.]))[:3]
    if not np.isfinite(point).all():
        return None
    return point, w*dw/width*z/fx, h*dh/height*z/fy


def plausible_bin(point, robot_xy, maximum_range=4.0):
    """Independent navigation guard against distant/background drop targets."""
    point = np.asarray(point)
    return (np.isfinite(point).all() and .60 < point[2] < 1.20 and
            .35 < np.linalg.norm(point[:2]-robot_xy) <= maximum_range)
