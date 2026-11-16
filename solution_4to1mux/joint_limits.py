"""URDF limits intersected with the live ERC interface's command envelope.

Live caps: ERC 2026 TIAGo Pro Interface Guide, sections 1 and 3. These caps
can only tighten the connected robot's URDF, never widen it. Grippers have
no checking-layer position margin. Units are metres or radians and seconds.
"""
from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np


def live_caps():
    caps = {
        'torso_lift_joint': (0.0, 0.35, 0.035),
        'head_1_joint': (*np.deg2rad([-75., 75.]), 0.5),
        'head_2_joint': (*np.deg2rad([-60., 20.]), 0.5),
        'gripper_left_finger_joint': (0.0, 0.803, None),
        'gripper_right_finger_joint': (0.0, 0.803, None),
    }
    for side in ('left', 'right'):
        degrees = [(-270, 30) if side == 'left' else (-30, 270),
                   (-140, 65), (-150, 150), (-140, 65),
                   (-90, 210) if side == 'left' else (-210, 90),
                   (-108, 172), (-150, 150)]
        for index, bounds in enumerate(degrees, 1):
            caps[f'arm_{side}_{index}_joint'] = (
                *np.deg2rad(bounds), 0.60 if index <= 2 else 0.75)
    return caps


LIVE_CAPS = live_caps()


@dataclass(frozen=True)
class JointLimit:
    lower: float
    upper: float
    velocity: float
    hard_lower: float
    hard_upper: float


class JointLimits:
    def __init__(self, urdf, profile='simulation'):
        if profile not in ('simulation', 'real'):
            raise ValueError(f'Unknown joint-limit profile: {profile}')
        self.by_name = {}
        root = ET.fromstring(urdf)
        for joint in root.findall('joint'):
            limit = joint.find('limit')
            if (limit is None or joint.find('mimic') is not None or
                    joint.get('type') not in ('revolute', 'prismatic')):
                continue
            name = joint.get('name')
            hard_lo, hard_hi = float(limit.get('lower')), float(limit.get('upper'))
            velocity = float(limit.get('velocity'))
            lo, hi = hard_lo, hard_hi
            safety = joint.find('safety_controller')
            if safety is not None:
                lo = max(lo, float(safety.get('soft_lower_limit', lo)))
                hi = min(hi, float(safety.get('soft_upper_limit', hi)))
            if profile == 'real' and name in LIVE_CAPS:
                cap_lo, cap_hi, speed = LIVE_CAPS[name]
                hard_lo, hard_hi = max(hard_lo, cap_lo), min(hard_hi, cap_hi)
                margin = 0.0 if name.startswith('gripper_') else 0.05
                lo, hi = max(lo, hard_lo+margin), min(hi, hard_hi-margin)
                if speed is not None:
                    velocity = min(velocity, speed)
            if (not np.isfinite([lo, hi, hard_lo, hard_hi, velocity]).all() or
                    lo >= hi or velocity <= 0):
                raise ValueError(f'Invalid or conflicting limits for {name}: [{lo}, {hi}], speed {velocity}')
            self.by_name[name] = JointLimit(lo, hi, velocity, hard_lo, hard_hi)

    def validate(self, names, positions):
        values = np.atleast_2d(np.asarray(positions, dtype=float))
        missing = [name for name in names if name not in self.by_name]
        if missing:
            raise ValueError(f'No URDF joint limits for {missing}')
        if (values.ndim != 2 or values.shape != (len(values), len(names)) or
                not len(values) or not np.isfinite(values).all()):
            raise ValueError('Joint command has missing or nonfinite positions.')
        for column, name in enumerate(names):
            bound = self.by_name[name]
            if (values[:, column].min() < bound.lower-1e-8 or
                    values[:, column].max() > bound.upper+1e-8):
                raise ValueError(f'{name} command outside [{bound.lower:.4f}, {bound.upper:.4f}]')

    def times(self, names, start, positions, minimum=0.0, peak_factor=1.0):
        """Cumulative segment times with a 30% speed reserve, checked at send time."""
        self.validate(names, positions)
        if not np.isfinite([minimum, peak_factor]).all() or minimum < 0 or peak_factor < 1:
            raise ValueError('Trajectory timing parameters must be finite with a peak factor >= 1.')
        start = np.asarray(start, dtype=float)
        if start.shape != (len(names),) or not np.isfinite(start).all():
            raise ValueError('Fresh finite joint positions are required to time a command.')
        velocity = np.array([self.by_name[name].velocity for name in names])
        points = np.atleast_2d(np.asarray(positions, dtype=float))
        times, elapsed = [], 0.0
        for target in points:
            elapsed += max(0.05, float(peak_factor*np.max(np.abs(target-start)/(0.7*velocity))))
            times.append(elapsed)
            start = target
        scale = max(1.0, minimum/elapsed)
        return np.asarray(times)*scale

    def clamp(self, names, positions):
        """For geometric head goals only; arm plans must be solved within bounds."""
        return [float(np.clip(value, self.by_name[name].lower, self.by_name[name].upper))
                for name, value in zip(names, positions)]
