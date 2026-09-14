import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'solution_4to1mux'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files must be installed or `ros2 launch` can't find them,
        # even though they exist in the source tree.
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Digit templates for the shelf-number detector. Without this the
        # detector silently falls back to synthetic templates, which match
        # noticeably worse than real crops from the sim.
        (os.path.join('share', package_name, 'templates'), glob('templates/*.png')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='ERC 2026 Library Assistant Robot solution.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'perception_node = solution_4to1mux.perception_node:main',
            'navigation_node = solution_4to1mux.navigation_node:main',
            'manipulation_node = solution_4to1mux.manipulation_node:main',
        ],
    },
)