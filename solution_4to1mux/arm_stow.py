"""Compute compact arm poses from URDF geometry and measured joint positions.

Clearance checks use joint-to-joint capsules and finite torso bounds, not
mesh collision checking. Planning fails if no checked straight joint path exists.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .geometry import ArmKinematics
from .torso_clearance import TorsoClearance


@dataclass(frozen=True)
class StowPlan:
    targets: dict
    duration: float


class ArmStowPlanner:
    """Find nearby compact poses for both arms while keeping the torso fixed."""

    def __init__(self, urdf, base, right_tip, profile='simulation'):
        self.torso = TorsoClearance(urdf)
        self.models = {}
        for side in ('left', 'right'):
            tip = right_tip.replace('right', side)
            model = ArmKinematics(urdf, base=base, tip=tip, profile=profile)
            # Include the gripper length, not just the wrist, in the envelope.
            model.grasp_transform = model.fixed_transform(
                tip, f'gripper_{side}_grasping_link')
            expected = ['torso_lift_joint']+[f'arm_{side}_{i}_joint' for i in range(1, 8)]
            if model.names != expected:
                raise ValueError(f'Unexpected {side} arm chain: {model.names}')
            self.models[side] = model

    @staticmethod
    def points(model, q):
        return model.arm_points(q, model.grasp_transform)

    def plan(self, joints, normal, wall_distance, dock_distance, *,
             front_limit=0.40, half_width=0.55, link_radius=0.06, clearance=0.08):
        normal = np.asarray(normal, dtype=float)
        if (normal.shape != (2,) or not np.isfinite(normal).all() or
                not np.isclose(np.linalg.norm(normal), 1.0) or
                not np.isfinite([wall_distance, dock_distance]).all()):
            raise ValueError('Stow planning needs a finite unit shelf normal and distances.')
        dimensions = [front_limit, half_width, link_radius, clearance]
        if not np.isfinite(dimensions).all() or min(dimensions) <= 0:
            raise ValueError('Stow envelope dimensions must be positive.')
        targets = {}
        duration = 1.0
        for side, model in self.models.items():
            current = np.array([joints[name] for name in model.names], dtype=float)
            if not np.isfinite(current).all():
                raise ValueError(f'Nonfinite {side} joint feedback.')
            if np.any(current < model.hard_lower-0.02) or np.any(current > model.hard_upper+0.02):
                raise ValueError(f'{side} arm feedback is outside URDF limits; refusing stow.')
            seed = np.clip(current, model.lower, model.upper)
            # Stow commands only the arms. Model the measured torso even when
            # it lies outside the live checking layer's command-margin band.
            seed[0] = current[0]
            shoulder = self.points(model, seed)[0]
            torso_boxes = self.torso.bounds(model, current)
            sign = 1.0 if shoulder[1] >= 0 else -1.0
            # Geometric intent: tuck the hand beside and below its own shoulder.
            # The optimizer determines every joint angle from the actual model.
            hand_goal = shoulder+np.array([0.0, sign*0.18, -0.25])

            def obstacle_margins(points, distance):
                # These are physical obstacles and must be clear even before
                # stowing. The higher hand/floor envelope below is a desired
                # transport posture, not the ground plane.
                return np.concatenate([
                    distance-clearance-link_radius-points[:, :2] @ normal,
                    points[:, 2]-link_radius-clearance])

            def posture_margins(points):
                # A hand in front of the torso may safely cross the centreline.
                # Check finite elbow/forearm/hand segments against the body,
                # not a side plane that extends all the way to the shelf.
                return self.torso.segment_margins(points[3:], torso_boxes, link_radius)

            def margins(q, compact=False, points=None):
                if points is None:
                    points = self.points(model, q)
                # All links, wrist and fingertips stay on the robot side of the
                # shelf; a capsule's centreline extrema occur at its endpoints.
                distance = dock_distance if compact else wall_distance
                result = [obstacle_margins(points, distance), posture_margins(points)]
                if compact:
                    result.extend([front_limit-link_radius-points[:, 0],
                                   points[:, 0]+0.36-link_radius,
                                   half_width-link_radius-np.abs(points[:, 1]),
                                   points[:, 2]-link_radius-0.30,
                                   shoulder[2]+0.28-points[:, 2]-link_radius])
                return np.concatenate(result)

            start_points = self.points(model, current)
            start_obstacles = obstacle_margins(start_points, wall_distance)
            if np.min(start_obstacles) < -0.003:
                count = len(start_points)
                index = int(np.argmin(start_obstacles))
                obstacle = 'shelf' if index < count else 'ground'
                raise ValueError(f'{side} arm has insufficient {obstacle} clearance '
                                 f'at the starting pose ({start_obstacles[index]:.3f} m margin); '
                                 'back away or reposition it first.')
            start_posture = posture_margins(start_points)
            if np.min(start_posture) < -0.015:
                raise ValueError(f'{side} arm has insufficient finite torso clearance '
                                 f'at the starting pose ({np.min(start_posture):.3f} m margin); '
                                 'reposition it before stowing.')
            # An extended arm need not already satisfy the transport envelope:
            # the default low-torso pose has fingertips below the transport
            # height. Check actual ground clearance throughout the path instead.
            # Small overlap with conservative torso bounding boxes may exist;
            # do not worsen it while retracting, and require
            # the endpoint to clear the entire compact envelope. Shelf/ground
            # margins are never relaxed to the starting posture.
            baseline = np.r_[np.zeros_like(start_obstacles),
                             np.minimum(start_posture, 0.0)]
            if (np.allclose(seed, current, atol=1e-8, rtol=0.0) and
                    np.min(margins(seed, compact=True)) >= -0.002):
                targets[side] = seed
                continue

            def expand(values):
                return np.r_[seed[0], values]

            path_fractions = [0.25, 0.50, 0.75]

            def residual(values):
                q = expand(values)
                points = self.points(model, q)
                return np.concatenate([
                    points[-1]-hand_goal,
                    50.0*np.minimum(margins(q, compact=True, points=points), 0.0),
                    *[50.0*np.minimum(margins(current+a*(q-current))-baseline, 0.0)
                      for a in path_fractions],
                    0.025*(q[1:]-seed[1:])/model.velocity[1:]])

            candidates = []
            rng = np.random.default_rng(6)
            starts = [seed[1:], (model.lower[1:]+model.upper[1:])/2]
            starts += list(rng.uniform(model.lower[1:], model.upper[1:], size=(4, 7)))
            best_end = best_path = -float('inf')
            for initial in starts:
                for _ in range(4):
                    solution = least_squares(
                        residual, initial, bounds=(model.lower[1:], model.upper[1:]),
                        max_nfev=160, ftol=1e-6, xtol=1e-6, gtol=1e-6)
                    q = expand(solution.x)
                    endpoint_margin = float(np.min(margins(q, compact=True)))
                    best_end = max(best_end, endpoint_margin)
                    if endpoint_margin < -0.002:
                        break
                    # Check the complete interpolated path, not only the endpoint.
                    samples = max(25, int(np.ceil(np.max(np.abs(q-seed))/0.04))+1)
                    swept = [(float(np.min(margins(current+a*(q-current))-baseline)), a)
                             for a in np.linspace(0, 1, samples)]
                    path_margin, fraction = min(swept)
                    best_path = max(best_path, path_margin)
                    if path_margin >= -0.003:
                        # Cubic interpolation peaks at 1.5 times average speed.
                        candidates.append((1.5*model.duration(current, q), q))
                        break
                    # Refine at the actual worst point instead of accepting a
                    # trajectory that passes only the coarse optimization samples.
                    path_fractions.append(fraction)
                    initial = solution.x
                if candidates:
                    break
            if not candidates:
                raise ValueError(f'No compact {side} arm pose with a clear stow path '
                                 f'(endpoint margin {best_end:.3f} m, path margin {best_path:.3f} m); refusing shelf docking.')
            travel, targets[side] = min(candidates, key=lambda item: item[0])
            duration = max(duration, travel)
        return StowPlan(targets, duration)
