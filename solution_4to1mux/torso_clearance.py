"""Finite URDF torso bounds for conservative arm-capsule clearance checks.

STL meshes are reduced to boxes, not used for exact mesh collision checking.
Only elbow/forearm/hand segments are checked; proximal shoulder attachments
are excluded. A finite shoulder-derived box is used for joint-only URDFs.
"""
from itertools import product
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np

from .geometry import _joint_origin

FACE_I, FACE_J = np.triu_indices(6, 1)


def _mesh_bounds(uri):
    if uri.startswith('package://'):
        from ament_index_python.packages import get_package_share_directory
        package, relative = uri[len('package://'):].split('/', 1)
        path = Path(get_package_share_directory(package))/relative
    else:
        path = Path(uri.removeprefix('file://'))
    raw = path.read_bytes()
    count = struct.unpack_from('<I', raw, 80)[0] if len(raw) >= 84 else 0
    if len(raw) == 84+50*count:
        dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)),
                          ('attribute', '<u2')])
        vertices = np.frombuffer(raw, dtype=dtype, count=count, offset=84)['vertices'].reshape(-1, 3)
    else:
        vertices = np.array([np.fromstring(line.strip()[7:], sep=' ')
                             for line in raw.decode().splitlines()
                             if line.strip().startswith('vertex ')])
    if vertices.size == 0 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError(f'Invalid torso STL: {path}')
    return vertices.min(axis=0), vertices.max(axis=0)


class TorsoClearance:
    def __init__(self, urdf):
        self.boxes = []
        self.error = ''
        root = ET.fromstring(urdf)
        try:
            for link in root.findall('link'):
                if link.get('name') not in ('torso_base_link', 'torso_lift_link'):
                    continue
                for collision in link.findall('collision'):
                    shape = collision.find('geometry')
                    if shape.find('box') is not None:
                        half = np.fromstring(shape.find('box').get('size'), sep=' ')/2
                        lower, upper = -half, half
                    elif shape.find('cylinder') is not None:
                        cylinder = shape.find('cylinder')
                        half = np.array([float(cylinder.get('radius'))]*2+
                                        [float(cylinder.get('length'))/2])
                        lower, upper = -half, half
                    elif shape.find('sphere') is not None:
                        half = np.full(3, float(shape.find('sphere').get('radius')))
                        lower, upper = -half, half
                    elif shape.find('mesh') is not None:
                        mesh = shape.find('mesh')
                        lower, upper = _mesh_bounds(mesh.get('filename'))
                        scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                        lower, upper = np.minimum(lower*scale, upper*scale), np.maximum(lower*scale, upper*scale)
                    else:
                        raise ValueError('Unsupported torso collision geometry.')
                    if (lower.shape != (3,) or upper.shape != (3,) or
                            not np.isfinite([lower, upper]).all() or np.any(lower >= upper)):
                        raise ValueError('Invalid torso collision bounds.')
                    self.boxes.append((link.get('name'), _joint_origin(collision), lower, upper))
        except (OSError, ImportError, LookupError, ValueError, AttributeError, TypeError) as error:
            # Incomplete mesh packages must not restore the old infinite wall.
            # Report the fallback explicitly through the navigation model log.
            self.error = str(error)
            self.boxes = []

    @property
    def source(self):
        return 'URDF torso collision bounds' if self.boxes else 'finite shoulder-derived torso envelope'

    def bounds(self, model, q):
        boxes = []
        for link, origin, lower, upper in self.boxes:
            transform = model.frame_transform(q, link) @ origin
            corners = np.array(list(product(*zip(lower, upper))))
            corners = corners @ transform[:3, :3].T+transform[:3, 3]
            boxes.append((corners.min(axis=0), corners.max(axis=0)))
        if not boxes:
            shoulder = model.arm_points(q)[0]
            # Finite, deliberately conservative fallback for stripped URDFs.
            boxes = [(np.array([shoulder[0]-.35, -abs(shoulder[1])-.045, .29]),
                      np.array([shoulder[0]+.22, abs(shoulder[1])+.045, shoulder[2]+.18]))]
        return boxes

    @staticmethod
    def segment_margins(points, boxes, radius):
        """Exact segment vs expanded-box margin (conservative at box corners).

        The box signed margin is the maximum of six affine face distances.
        Its minimum along a segment is at an endpoint or face intersection.
        Checking only joint endpoints would miss a link passing through torso.
        """
        starts, deltas = points[:-1], np.diff(points, axis=0)
        lower, upper = np.asarray([box[0] for box in boxes]), np.asarray([box[1] for box in boxes])
        intercept = np.concatenate([starts[None]-upper[:, None], lower[:, None]-starts[None]], axis=2)-radius
        slope = np.concatenate([deltas, -deltas], axis=1)
        denominator = slope[:, FACE_I]-slope[:, FACE_J]
        numerator = intercept[:, :, FACE_J]-intercept[:, :, FACE_I]
        fractions = np.divide(numerator, denominator[None], out=np.zeros_like(numerator),
                              where=np.abs(denominator[None]) > 1e-12)
        fractions = np.concatenate([np.zeros((*intercept.shape[:2], 1)),
                                    np.ones((*intercept.shape[:2], 1)),
                                    np.clip(fractions, 0.0, 1.0)], axis=2)
        values = np.max(intercept[:, :, None]+fractions[:, :, :, None]*slope[None, :, None], axis=3)
        return values.min(axis=2).ravel()
