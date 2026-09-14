"""ROS-independent geometry shared by perception, navigation and arm control."""

import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * cross @ cross


def quaternion_matrix(q):
    v = np.array([q.x, q.y, q.z, q.w], dtype=float)
    v /= max(np.linalg.norm(v), 1e-12)
    x, y, z, w = v
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def transform_matrix(transform):
    result = np.eye(4)
    result[:3, :3] = quaternion_matrix(transform.rotation)
    t = transform.translation
    result[:3, 3] = [t.x, t.y, t.z]
    return result


def pose_xyz(pose):
    p = pose.pose.position
    return np.array([p.x, p.y, p.z])


def yaw(q):
    r = quaternion_matrix(q)
    return math.atan2(r[1, 0], r[0, 0])


def depth_median(depth, px, py, radius=1):
    """Reject NaN/inf/zero instead of publishing a fabricated distance."""
    h, w = depth.shape
    x, y = int(round(px)), int(round(py))
    if not (0 <= x < w and 0 <= y < h):
        return None
    roi = depth[max(0, y-radius):min(h, y+radius+1),
                max(0, x-radius):min(w, x+radius+1)]
    values = roi[np.isfinite(roi) & (roi > 0.15) & (roi < 8.0)]
    return float(np.median(values)) if values.size else None


def row_for_height(marker_z, book_z, top_offset=0.665, spacing=0.33):
    index = int(round((marker_z - top_offset - book_z) / spacing))
    if not 0 <= index < 4:
        return None
    expected = marker_z - top_offset - index * spacing
    return index + 1 if abs(expected - book_z) < 0.14 else None


def body_velocity(dx, dy, heading, maximum, gain=1.2):
    """Limit vector magnitude (including diagonal driving), not each axis."""
    distance = math.hypot(dx, dy)
    if distance < 1e-9:
        return 0.0, 0.0
    speed = min(maximum, gain * distance)
    c, s = math.cos(heading), math.sin(heading)
    return speed * (c*dx+s*dy)/distance, speed * (-s*dx+c*dy)/distance


def forward_command(dx, dy, heading, final_heading, linear_cap, angular_cap,
                    tolerance=0.04):
    """Face the travel direction, drive forward, then set the docking heading.

    Long travel never commands lateral velocity. A separate low-speed docking
    mode can keep the camera facing the shelf during the last small alignment.
    """
    distance = math.hypot(dx, dy)
    desired = math.atan2(dy, dx) if distance > tolerance else final_heading
    error = wrap(desired-heading)
    if distance <= tolerance and abs(error) <= 0.04:
        return 0.0, 0.0, 0.0, True
    omega = float(np.clip(2.0*error, -angular_cap, angular_cap))
    forward = 0.0
    if distance > tolerance and abs(error) < 0.18:
        forward = min(linear_cap, 1.8*distance)*math.cos(error)
    return forward, 0.0, omega, False


def scan_points(ranges, angle_min, angle_increment, range_min, range_max, transform):
    """Laser frame -> base frame; the ERC front sensor is yawed -45 degrees."""
    r = np.asarray(ranges, dtype=float)
    a = angle_min + np.arange(r.size) * angle_increment
    valid = np.isfinite(r) & (r >= range_min) & (r <= range_max)
    points = np.stack([r[valid]*np.cos(a[valid]), r[valid]*np.sin(a[valid]),
                       np.zeros(valid.sum()), np.ones(valid.sum())])
    return (transform @ points)[:2].T


class ArmKinematics:
    """Position/orientation IK from the actual URDF, with hard joint limits.

    This is a local waypoint solver, not a collision-aware MoveIt planner.
    The caller must approach the shelf square-on and keep waypoints clear.
    """

    def __init__(self, urdf, base='base_footprint', tip='gripper_right_grasping_link'):
        root = ET.fromstring(urdf)
        by_child = {j.find('child').get('link'): j for j in root.findall('joint')}
        chain, current = [], tip
        while current != base:
            if current not in by_child:
                raise ValueError(f'No URDF chain from {base} to {tip}')
            joint = by_child[current]
            chain.append(joint)
            current = joint.find('parent').get('link')
        self.chain, self.names, self.lower, self.upper, self.velocity = [], [], [], [], []
        for j in reversed(chain):
            origin = np.eye(4)
            o = j.find('origin')
            if o is not None:
                origin[:3, 3] = np.fromstring(o.get('xyz', '0 0 0'), sep=' ')
                r, p, y = np.fromstring(o.get('rpy', '0 0 0'), sep=' ')
                origin[:3, :3] = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
            kind = j.get('type')
            axis = np.fromstring(j.find('axis').get('xyz'), sep=' ') if j.find('axis') is not None else np.array([1., 0., 0.])
            index = None
            if kind != 'fixed':
                index = len(self.names)
                self.names.append(j.get('name'))
                limit = j.find('limit')
                safety = j.find('safety_controller')
                lo, hi = float(limit.get('lower')), float(limit.get('upper'))
                if safety is not None:
                    lo = max(lo, float(safety.get('soft_lower_limit', lo)))
                    hi = min(hi, float(safety.get('soft_upper_limit', hi)))
                self.lower.append(lo)
                self.upper.append(hi)
                self.velocity.append(float(limit.get('velocity')))
            self.chain.append((origin, kind, axis, index))
        self.lower, self.upper, self.velocity = map(np.array, (self.lower, self.upper, self.velocity))

    def fk(self, joints):
        result = np.eye(4)
        for origin, kind, axis, index in self.chain:
            result = result @ origin
            if index is not None:
                motion = np.eye(4)
                if kind == 'prismatic':
                    motion[:3, 3] = axis * joints[index]
                else:
                    motion[:3, :3] = rotation(axis, joints[index])
                result = result @ motion
        return result

    def solve(self, xyz, orientation, seed, attempts=12, iterations=200):
        target = np.asarray(xyz, dtype=float)
        seed = np.clip(np.asarray(seed, dtype=float), self.lower, self.upper)
        rng = np.random.default_rng(6)
        best = float('inf')

        def residual(q):
            actual = self.fk(q)
            # Full matrix residual avoids a false zero at a 180-degree rotation.
            return np.r_[actual[:3, 3]-target,
                         0.3*(actual[:3, :3]-orientation).ravel()]

        for attempt in range(attempts):
            initial = seed if attempt == 0 else rng.uniform(self.lower, self.upper)
            result = least_squares(residual, initial, bounds=(self.lower, self.upper),
                                   max_nfev=iterations, ftol=1e-7, xtol=1e-7, gtol=1e-7)
            actual = self.fk(result.x)
            pos_error = float(np.linalg.norm(actual[:3, 3]-target))
            best = min(best, pos_error)
            if pos_error < 0.008 and np.linalg.norm(actual[:3, :3]-orientation) < 0.06:
                return result.x
        raise ValueError(f'IK failed for {target.round(3).tolist()} (best position error {best:.3f} m)')

    def duration(self, start, end, minimum=1.0):
        # Torso limit is only 0.035 m/s: 0.35 m cannot be commanded in 2 seconds.
        return max(minimum, float(np.max(np.abs(np.asarray(end)-start)/(0.7*self.velocity))))
