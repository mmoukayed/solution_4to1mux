"""One build identity for all nodes and the installation verifier."""
import hashlib
from pathlib import Path

BUILD = '8.0'
VERSION = '0.8.0'
CORE_FILES = ('alignment.py', 'build_info.py', 'runtime_health.py', 'ros_topics.py',
              'geometry.py', 'digit_detector.py', 'perception_node.py',
              'navigation_node.py', 'manipulation_node.py')


def source_fingerprint(directory=None):
    """Identify the actual loaded Python source, not a possibly stale README."""
    directory = Path(directory) if directory is not None else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in CORE_FILES:
        digest.update(name.encode())
        digest.update((directory/name).read_bytes())
    return digest.hexdigest()[:16]


def build_banner():
    return f'BUILD {BUILD}; version={VERSION}; source={source_fingerprint()}; path={Path(__file__).resolve().parent}'
