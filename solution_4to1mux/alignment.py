"""ROS-independent shelf fitting, joint limits and coordinated motion defaults."""
import math
import xml.etree.ElementTree as ET

import numpy as np


MOTION_DEFAULTS = {
    'linear_speed': 0.45, 'strafe_speed': 0.16, 'rotate_speed': 0.50,
    'search_rotate_speed': 0.22, 'docking_speed': 0.16,
    'carry_speed': 0.25, 'grasp_return_speed': 0.08,
    'linear_acceleration': 0.35, 'angular_acceleration': 0.8,
    'shelf_alignment_strafe_speed': 0.12,
    'arm_motion_speed': 0.40, 'arm_feedback_tolerance': 0.10,
    'arm_feedback_settle': 0.25, 'arm_tuck_timeout': 25.0,
}


def angle_delta(target, current):
    return math.atan2(math.sin(target-current), math.cos(target-current))


def fit_shelf_line(points, camera_xy, min_span=0.5, residual=0.06):
    """Fit a shelf front to marker centres; reject non-collinear outliers."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or len(points) < 2 or not np.isfinite(points).all():
        return None
    xy = points[:, :2]
    best = None
    for i in range(len(xy)):
        for j in range(i+1, len(xy)):
            tangent = xy[j]-xy[i]
            span = np.linalg.norm(tangent)
            if span < min_span:
                continue
            tangent /= span
            normal = np.array([-tangent[1], tangent[0]])
            errors = np.abs((xy-xy[i]) @ normal)
            inliers = errors <= residual
            score = (int(inliers.sum()), -float(np.median(errors[inliers])), span)
            if best is None or score > best[0]:
                best = (score, inliers)
    if best is None or best[0][0] < max(2, math.ceil(0.6*len(points))):
        return None
    inliers = points[best[1]]
    centre = np.median(inliers, axis=0)
    _, _, axes = np.linalg.svd(inliers[:, :2]-inliers[:, :2].mean(axis=0), full_matrices=False)
    tangent = axes[0]
    if np.ptp(inliers[:, :2] @ tangent) < min_span:
        return None
    normal = np.array([-tangent[1], tangent[0]])
    if normal @ (centre[:2]-camera_xy) < 0:
        normal = -normal
    return centre, math.atan2(normal[1], normal[0]), best[1]


def read_joint_limits(urdf):
    """Match the live checker: URDF/soft bounds and 0.06 hard-limit margin."""
    result = {}
    for joint in ET.fromstring(urdf).findall('joint'):
        limit = joint.find('limit')
        if limit is None or limit.get('lower') is None or limit.get('upper') is None:
            continue
        margin = 0.06 if joint.get('name', '').startswith('arm_') else 0.0
        lo, hi = float(limit.get('lower'))+margin, float(limit.get('upper'))-margin
        safety = joint.find('safety_controller')
        if safety is not None:
            lo = max(lo, float(safety.get('soft_lower_limit', lo)))
            hi = min(hi, float(safety.get('soft_upper_limit', hi)))
        velocity = float(limit.get('velocity'))
        if not np.isfinite([lo, hi, velocity]).all() or lo >= hi or velocity <= 0:
            raise ValueError(f'Invalid joint bounds for {joint.get("name")}')
        result[joint.get('name')] = (lo, hi, velocity)
    return result
