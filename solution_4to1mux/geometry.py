"""ROS-independent geometry shared by perception, navigation and arm control."""

import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares

from .ros_topics import FRAMES
from .joint_limits import JointLimits


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


def _joint_origin(joint):
    """Return a URDF joint's parent-to-child origin transform."""
    result = np.eye(4)
    origin = joint.find('origin')
    if origin is not None:
        result[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        roll, pitch, yaw_angle = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
        result[:3, :3] = (rotation([0, 0, 1], yaw_angle)
                          @ rotation([0, 1, 0], pitch)
                          @ rotation([1, 0, 0], roll))
    return result


class ArmKinematics:
    """Position/orientation IK from the actual URDF, with hard joint limits.

    This is a local waypoint solver, not a collision-aware MoveIt planner.
    The caller must approach the shelf square-on and keep waypoints clear.
    """

    def __init__(self, urdf, base=FRAMES['base'], tip=FRAMES['arm_tip'], profile='simulation'):
        root = ET.fromstring(urdf)
        self.limits = JointLimits(urdf, profile)
        by_child = {j.find('child').get('link'): j for j in root.findall('joint')}
        self.by_child = by_child
        self.base_link = base
        self.tip_link = tip
        chain, current = [], tip
        while current != base:
            if current not in by_child:
                raise ValueError(f'No URDF chain from {base} to {tip}')
            joint = by_child[current]
            chain.append(joint)
            current = joint.find('parent').get('link')
        self.chain, self.names, self.lower, self.upper, self.velocity = [], [], [], [], []
        self.hard_lower, self.hard_upper = [], []
        for j in reversed(chain):
            origin = _joint_origin(j)
            kind = j.get('type')
            axis = np.fromstring(j.find('axis').get('xyz'), sep=' ') if j.find('axis') is not None else np.array([1., 0., 0.])
            index = None
            if kind != 'fixed':
                index = len(self.names)
                self.names.append(j.get('name'))
                bound = self.limits.by_name[j.get('name')]
                self.hard_lower.append(bound.hard_lower)
                self.hard_upper.append(bound.hard_upper)
                self.lower.append(bound.lower)
                self.upper.append(bound.upper)
                self.velocity.append(bound.velocity)
            self.chain.append((origin, kind, axis, index))
        self.lower, self.upper, self.velocity = map(np.array, (self.lower, self.upper, self.velocity))
        self.hard_lower, self.hard_upper = map(np.array, (self.hard_lower, self.hard_upper))

    def fixed_transform(self, parent, child):
        """Get a fixed URDF transform between links outside the solved chain."""
        transforms = []
        current = child
        while current != parent:
            joint = self.by_child.get(current)
            if joint is None or joint.get('type') != 'fixed':
                raise ValueError(f'No fixed URDF transform from {parent} to {child}')
            transforms.append(_joint_origin(joint))
            current = joint.find('parent').get('link')
        result = np.eye(4)
        for transform in reversed(transforms):
            result = result @ transform
        return result

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

    def arm_points(self, joints, tip_transform=None):
        """Return arm joint centres and the tip for conservative clearance checks."""
        result = np.eye(4)
        points = []
        for origin, kind, axis, index in self.chain:
            result = result @ origin
            if index is not None:
                motion = np.eye(4)
                if kind == 'prismatic':
                    motion[:3, 3] = axis*joints[index]
                else:
                    points.append(result[:3, 3].copy())
                    motion[:3, :3] = rotation(axis, joints[index])
                result = result @ motion
        points.append(result[:3, 3].copy())
        if tip_transform is not None:
            points.append((result @ tip_transform)[:3, 3])
        return np.vstack(points)

    def solve(self, xyz, orientation, seed, attempts=12, iterations=200, fixed=None,
              max_step=None):
        target = np.asarray(xyz, dtype=float)
        seed = np.clip(np.asarray(seed, dtype=float), self.lower, self.upper)
        fixed = {int(index): float(value) for index, value in (fixed or {}).items()}
        if any(index < 0 or index >= len(seed) for index in fixed):
            raise ValueError('Fixed joint index is outside the kinematic chain.')
        if any(not self.lower[index] <= value <= self.upper[index]
               for index, value in fixed.items()):
            raise ValueError('A fixed joint value is outside its URDF limits.')
        anchor = seed.copy()
        for index, value in fixed.items():
            anchor[index] = value
        free = np.array([index for index in range(len(seed)) if index not in fixed], dtype=int)
        lower, upper = self.lower.copy(), self.upper.copy()
        if max_step is not None:
            step = np.broadcast_to(np.asarray(max_step, dtype=float), seed.shape)
            if not np.isfinite(step).all() or np.min(step) <= 0:
                raise ValueError('Local IK step bounds must be positive and finite.')
            lower = np.maximum(lower, anchor-step)
            upper = np.minimum(upper, anchor+step)
            attempts = 1  # Never jump to a random global branch near a book.
        rng = np.random.default_rng(6)
        best = float('inf')

        def expand(values):
            q = anchor.copy()
            q[free] = values
            return q

        def residual(values):
            q = expand(values)
            actual = self.fk(q)
            # Full matrix residual avoids a false zero at a 180-degree rotation.
            error = np.r_[actual[:3, 3]-target,
                          0.3*(actual[:3, :3]-orientation).ravel()]
            if max_step is not None:
                error = np.r_[error, 0.001*(q[free]-anchor[free])/self.velocity[free]]
            return error

        for attempt in range(attempts):
            initial = anchor[free] if attempt == 0 else rng.uniform(lower[free], upper[free])
            result = least_squares(residual, initial,
                                   bounds=(lower[free], upper[free]),
                                   max_nfev=iterations, ftol=1e-7, xtol=1e-7, gtol=1e-7)
            solution = expand(result.x)
            actual = self.fk(solution)
            pos_error = float(np.linalg.norm(actual[:3, 3]-target))
            best = min(best, pos_error)
            if pos_error < 0.008 and np.linalg.norm(actual[:3, :3]-orientation) < 0.06:
                return solution
        raise ValueError(f'IK failed for {target.round(3).tolist()} (best position error {best:.3f} m)')

    def cartesian_path(self, xyz, orientation, seed, fixed=None, spacing=0.01):
        """Track a short straight tool path on the measured IK branch."""
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError('Cartesian path spacing must be positive and finite.')
        q = np.asarray(seed, dtype=float).copy()
        origin = self.fk(q)[:3, 3]
        target = np.asarray(xyz, dtype=float)
        count = max(1, int(np.ceil(np.linalg.norm(target-origin)/spacing)))
        path = []
        step = np.r_[0.01, np.full(len(q)-1, 0.35)]
        for fraction in np.linspace(0, 1, count+1)[1:]:
            q = self.solve(origin+fraction*(target-origin), orientation, q,
                           fixed=fixed, max_step=step, iterations=250)
            path.append(q)
        self.limits.validate(self.names, path)
        return np.asarray(path)

    def duration(self, start, end, minimum=1.0):
        # Torso limit is only 0.035 m/s: 0.35 m cannot be commanded in 2 seconds.
        return max(minimum, float(np.max(np.abs(np.asarray(end)-start)/(0.7*self.velocity))))
