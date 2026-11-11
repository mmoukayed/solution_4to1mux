#!/usr/bin/env python3
"""
Generate the MuJoCo-flavoured URDF for the ERC TIAGo Pro dev simulator.

Starts from the competition's pregenerated URDF (erc_description/urdf/
tiago_pro.urdf) so the robot the dev simulator runs is byte-derived from the
one the competition ships: same links, masses, joint limits, meshes and
ros2_control joint set. Applies only what MuJoCo needs:

  1. Strip every <gazebo> element PAL's xacro emits (plugins, sensors,
     friction tags) - MuJoCo reads none of them - along with the transmissions
     and the two laser housings (see strip_laser_housings).
  2. Swap the hardware plugin PAL's xacro emits for
     mujoco_ros2_control/MujocoSystemInterface reading the MJCF from a topic.
  3. Embed a <mujoco_inputs> block for the URDF->MJCF converter:
       - one MuJoCo <position> actuator per commanded joint, gains taken from
         PAL's own tiago_pro_mujoco pids.yaml (p -> kp, u_clamp -> forcerange)
       - <equality> joint couplings replacing the URDF <mimic> four-bar
         constraints that MJCF conversion drops
       - the head camera as a MuJoCo RGB-D camera on the optical frame,
         at the resolution and field of view the competition URDF states
The camera needs no frame of its own: the converter treats the named site as a
REP-103 optical frame and applies the optical->MuJoCo rotation itself.

Run inside the container after building:
    ros2 run erc_bringup generate_mujoco_urdf.py -o /tmp/erc_mujoco.urdf
"""

import argparse
import math
import os
import re
import sys

from ament_index_python.packages import get_package_share_directory

# PAL's own MuJoCo gains for the TIAGo Pro. kp and the force clamp come from
# tiago_pro_simulation (humble-devel, 1.19.0) tiago_pro_mujoco/config/pids.yaml
# (p -> kp, u_clamp -> forcerange); kv is PAL's own MuJoCo damping ladder from
# pal_sea_arm_description/mujoco/mj_tags.xacro, which is tuned per arm joint
# rather than derived. kv=None means PAL ships no damping for that joint, and
# the actuator falls back to critical damping.
#                                 kp        clamp    kv
PAL_GAINS = {
    'torso_lift_joint':           (10000.0, 2000.0, None),
    'head_1_joint':               (2000.0,  5.197,  None),
    'head_2_joint':               (2000.0,  2.77,   None),
    'arm_left_1_joint':           (1000.0,  43.0,   50.0),
    'arm_left_2_joint':           (1000.0,  43.0,   60.0),
    'arm_left_3_joint':           (1000.0,  26.0,   50.0),
    'arm_left_4_joint':           (500.0,   26.0,   20.0),
    'arm_left_5_joint':           (1000.0,  26.0,   20.0),
    'arm_left_6_joint':           (1000.0,  26.0,   20.0),
    'arm_left_7_joint':           (500.0,   26.0,   20.0),
    'arm_right_1_joint':          (1000.0,  43.0,   50.0),
    'arm_right_2_joint':          (1000.0,  43.0,   60.0),
    'arm_right_3_joint':          (1000.0,  26.0,   50.0),
    'arm_right_4_joint':          (500.0,   26.0,   20.0),
    'arm_right_5_joint':          (1000.0,  26.0,   20.0),
    'arm_right_6_joint':          (1000.0,  26.0,   20.0),
    'arm_right_7_joint':          (500.0,   26.0,   20.0),
    # Grippers: kp raised from PAL's 100 so the position servo can develop
    # the full 8 N force limit within the screw's travel; the clamp itself
    # stays PAL's number, so max pinch is unchanged in principle and the
    # grip force is decided by the force-targeted close in the pick script.
    'gripper_left_finger_joint':  (300.0,   8.0,    None),
    'gripper_right_finger_joint': (300.0,   8.0,    None),
}

# The four mecanum wheels, in the order the mecanum kinematics wants them:
# front-left, front-right, rear-left, rear-right. The kinematic constants that go
# with them - radius 0.0762 m and the yaw lever arm lx+ly = 0.46717 m, from hub
# positions (+-0.244, +-0.22317) measured off the compiled model - are consumed by
# mecanum_drive_controller and BaseVelocityPlugin, so they live in
# controller_params.yaml and mujoco_plugins.yaml rather than here.
WHEEL_JOINTS = ('wheel_front_left_joint', 'wheel_front_right_joint',
                'wheel_rear_left_joint', 'wheel_rear_right_joint')
# PAL's <limit effort="6.0"> on each wheel joint. Four wheels at that torque is
# 4 * 6.0 / 0.0762 = 315 N of traction and 315 * 0.46717 = 147 Nm of yaw torque;
# those are the caps mujoco_plugins.yaml puts on the base servo.
WHEEL_EFFORT = 6.0
# PAL ships <dynamics damping="1.0" friction="2.0"/> on the wheel joints. Those are
# motor-side numbers that were never referred through the gearbox: 2.0 Nm of dry
# friction is 26 N of drag per wheel before it will turn at all, and 1.0 Nm/(rad/s)
# of damping costs 6.6 Nm at 0.5 m/s -- more than the 6.0 Nm the joint is allowed to
# produce, so the wheel simply cannot reach its commanded speed and the encoders
# under-read everything above ~0.3 m/s. Replace them with bearing-scale values so the
# wheel speed, and therefore the odometry derived from it, is usable.
WHEEL_DAMPING = 0.01
WHEEL_FRICTIONLOSS = 0.0
# Reflected inertia of the hub motor and gearbox, which a URDF cannot express and
# PAL's therefore does not carry. Without it the wheel's own inertia is 4e-4 kg m2,
# far too light for a velocity servo to hold at this timestep: the servo rang, and
# one command in six came out 7% low with several percent of cross-axis coupling.
# Anything from 0.01 upwards removes it completely.
WHEEL_ARMATURE = 0.01
# Wheel velocity-servo gain. The base servo chases the mecanum forward kinematics of
# the wheels' MEASURED speed, so any lag between a wheel's command and its actual
# speed lands directly on the base: the lag is (ground drag torque)/kv, and at kv=1
# it cost 3-43% of the commanded twist. kv=20 with the friction below holds every
# axis within 0.7%; pushing either much further makes the contact solver ring.
WHEEL_KV = 20.0
# Tangential friction of the wheel geoms against the floor. The rollers are not
# modelled as geometry, so a wheel that gripped would fight the analytic traction
# instead of producing it -- the same reason the competition's Gazebo build zeroes
# mu2 perpendicular to the roller axis, except that here there is no roller axis to
# keep grip along, so all of it goes. The wheels still carry the robot's weight and
# still collide with the world; they just do not resist sliding.
WHEEL_FRICTION = '0.01 0.001 0.0001'
# The fingertip pads: rubber on a book's paper cover, nearer 1 than the
# competition's 2.7, and swept for the dev sim's friction band (ERC_PAD_MU).
# priority 3 puts the pad's friction in charge of every contact it is in -
# above the books' priority 2 (generate_mujoco_world.py), whose own friction
# then governs book on shelf and book on bin only. Without it one number
# governed both sides of a pinch and of a drag, and no drag can ever beat a
# shelf whose friction equals the pad's. The winning geom sets the contact's
# condim too, so the pad must be condim 6 itself: at the default 3 no pad
# contact carries torsional friction and the book swings about the grasp axis.
PAD_MU = float(os.environ.get('ERC_PAD_MU') or 1.0)
# How hard the pad is. solref sets a contact's stiffness as an acceleration,
# so its N/m scales with the effective mass at the contact, and at the default
# time constant (0.02 s) the 8 N pinch sank the pads 1-3 mm into the book, and
# up to 5 mm while the arm or the base accelerated - the screw closing
# 0.0275 -> 0.0225 on a 30 mm book - and the book swung loose in the jaws in
# those moments (issue #57). A rubber pad on a card cover gives a fraction of a
# millimetre: 2e5-1e6 N/m for 2 mm of rubber over 4 cm^2. 0.004 s is twice the
# timestep, the stiffest MuJoCo allows (refsafe clamps anything lower), and
# there the pinch sinks about 0.1 mm, still a few times softer than rubber
# (ERC_PAD_SOLREF). The pad's priority makes its solref govern every pad contact.
PAD_SOLREF = float(os.environ.get('ERC_PAD_SOLREF') or 0.004)

# The gripper is a four-bar linkage: the fingertip is pinned to BOTH the inner
# finger (a URDF joint) and the outer finger (a loop the URDF cannot express,
# so it fakes it with a <mimic> ratio). MuJoCo can close the loop for real, so
# the fingertip's <connect> replaces its mimic equality rather than joining it:
# pinch force then reacts through the outer finger as it does on the hardware,
# instead of being carried entirely by a scripted joint ratio.
# Anchors are in the fingertip body frame, from PAL's own call site
# (pal_sea_arm_description/robots/pal_sea_arm.urdf.xacro).
FOUR_BAR_ANCHOR = {'left': '0.014 -0.001 0', 'right': '0.014 -0.0015 0'}

# PAL's xacro emits one of several simulator hardware plugins depending on its
# own arguments, so match whichever one landed rather than a fixed name.
HARDWARE_PLUGIN_RE = re.compile(r'<plugin>[^<]*(?:GazeboSystem|GazeboSimSystem|'
                                r'MujocoSystem\w*)</plugin>')

# The two base SICK scanners, matched to what the robot description declares:
# 818 samples across 270 degrees at 10 Hz, 0.05-25 m. The lidar extension casts
# rays in the site's own frame - x is azimuth zero, z is the scan normal - and
# the converter emits one site per URDF link, so each scanner rides its own
# link frame.
LASERS = (
    ('scan_front_raw', 'base_front_laser_link'),
    ('scan_rear_raw', 'base_rear_laser_link'),
)
# Each scanner's specification is defined once, in the robot description's own
# gpu_lidar sensor block, and read back out by laser_spec() before
# strip_gazebo_blocks deletes it - the same contract the head camera uses - so a
# simulated scan cannot drift from the one the shipped URDF describes.

# The base IMU. mujoco_ros2_control builds a ros2_control IMU sensor out of
# three MJCF sensors whose names share a base and take these suffixes, and
# imu_sensor_broadcaster turns that into a sensor_msgs/Imu.
IMU_SENSOR = 'base_imu_sensor'
IMU_SITE = 'base_imu_link'

# The head camera. Its optics are defined once, in the robot description's own
# head camera sensor (generate_urdf.py retargets PAL's stock module to the
# competition's D435 there), and read back out here before strip_gazebo_blocks
# deletes the block - so the simulated camera cannot drift from the one the
# shipped URDF describes. Unlike the scanners above, nothing here is copied.
HEAD_CAMERA_LINK = 'head_front_camera_link'
D435_DEPTH_HFOV = 87.0      # deg: the RealSense D435's depth field of view across
D435_COLOUR_HFOV = 69.4     # deg: ...and its colour camera's (Intel's figure; the robot's calibration gives 70.1)


def strip_gazebo_blocks(urdf: str) -> str:
    n = urdf.count('<gazebo')
    urdf = re.sub(r'[ \t]*<gazebo( [^>]*)?>.*?</gazebo>\n?', '', urdf, flags=re.S)
    if '<gazebo' in urdf:
        sys.exit('ERROR: gazebo blocks survived stripping')
    print(f'stripped {n} gazebo blocks')
    return urdf


def strip_laser_housings(urdf: str) -> str:
    """Remove the scanners' own housing geometry.

    The lidar extension casts its rays with no self-exclusion, and mj_multiRay
    filters by geom group and the static flag - not by contype/conaffinity - so
    a housing left in the model is hit by its own scanner whatever its collision
    flags say. Measured: every one of the 818 rays returned the housing mesh at
    20.6 mm. Both the visual and the collision element have to go, because the
    converter emits a collision geom from the visual mesh as well.

    The housings are 8 cm nubs sitting inside the base's own collision envelope,
    so the robot's contact behaviour is unchanged; they simply stop being drawn.
    """
    n = 0
    for _name, site in LASERS:
        m = re.search(r'(<link name="' + re.escape(site) + r'"[^>]*>)(.*?)(</link>)',
                      urdf, re.S)
        if not m:
            # A legitimate configuration: generate_urdf.py --laser_model
            # no-laser produces a robot with no scanners at all, and the LASERS
            # loop below skips them for the same reason.
            print(f'note: no <link name="{site}">; this robot has no scanner there')
            continue
        body, count = re.subn(r'[ \t]*<(visual|collision)(?:\s[^>]*)?>.*?</\1>\n?',
                              '', m.group(2), flags=re.S)
        # Half-stripping is worse than not stripping: whatever is left is
        # opaque to the scanner's own rays and every one of the 818 samples
        # comes back as the housing, which looks like a plausible scan.
        if count != 2:
            sys.exit(f'ERROR: expected one <visual> and one <collision> in '
                     f'{site}, removed {count}. Refusing to ship a scanner '
                     f'that would read its own housing.')
        urdf = urdf[:m.start()] + m.group(1) + body + m.group(3) + urdf[m.end():]
        n += count
    print(f'stripped {n} laser housing visual/collision elements')
    return urdf


def command_wheels(urdf: str) -> str:
    """Give the wheel joints a velocity command interface.

    generate_urdf.py leaves them state-only because the competition's Gazebo build
    drives them from the gz mecanum plugin, outside ros2_control. Here
    mecanum_drive_controller commands them through ros2_control like any other joint,
    so each needs a command interface to claim -- and a matching MuJoCo <velocity>
    actuator, added in build_mujoco_inputs.
    """
    n = 0
    for joint in WHEEL_JOINTS:
        state_only = (f'<joint name="{joint}">\n'
                      f'      <state_interface name="position"/>\n')
        if state_only not in urdf:
            if f'name="{joint}"' in urdf:
                sys.exit(f'ERROR: {joint} is in the description but not as a '
                         f'state-only <ros2_control> joint, so it cannot be '
                         f'given a command interface. mecanum_drive_controller '
                         f'would fail to activate and the base would be dead.')
            print(f'note: no {joint}; this robot has no wheel there')
            continue
        urdf = urdf.replace(state_only,
                            f'<joint name="{joint}">\n'
                            f'      <command_interface name="velocity"/>\n'
                            f'      <state_interface name="position"/>\n', 1)
        n += 1
    if n and n != len(WHEEL_JOINTS):
        sys.exit(f'ERROR: gave {n} of {len(WHEEL_JOINTS)} wheels a command '
                 f'interface. A half-commanded base would drive on some wheels '
                 f'and drag on the others.')
    print(f'{n} wheels given a velocity command interface')
    return urdf


def strip_transmissions(urdf: str) -> str:
    # mujoco_ros2_control demands a matching MuJoCo actuator for every
    # transmission actuator, so leaving these in would mean inventing actuators
    # that drive nothing. Every joint that is actually commanded gets a
    # <position> actuator on the joint itself instead, which bypasses the
    # transmission - including the gripper, whose PalGripperTransmission is the
    # one block here with a reduction other than 1.0.
    n = urdf.count('<transmission')
    urdf = re.sub(r'[ \t]*<transmission( [^>]*)?>.*?</transmission>\n?', '',
                  urdf, flags=re.S)
    print(f'stripped {n} transmissions')
    return urdf


def joint_limits(urdf: str):
    """name -> (lower, upper) for every joint with a <limit>."""
    lims = {}
    for m in re.finditer(r'<joint name="([^"]+)" type="[^"]+">(.*?)</joint>', urdf, re.S):
        lm = re.search(r'<limit[^>]*lower="([^"]+)"[^>]*upper="([^"]+)"', m.group(2))
        if lm:
            lims[m.group(1)] = (float(lm.group(1)), float(lm.group(2)))
    return lims


def mimic_table(urdf: str):
    """(mimic_joint, driven_joint, multiplier) parsed from ros2_control params."""
    block = urdf[urdf.index('<ros2_control'):urdf.index('</ros2_control>')]
    out = []
    for m in re.finditer(r'<joint name="([^"]+)">(.*?)</joint>', block, re.S):
        mm = re.search(r'<param name="mimic">([^<]+)</param>', m.group(2))
        mult = re.search(r'<param name="multiplier">([^<]+)</param>', m.group(2))
        if mm:
            out.append((m.group(1), mm.group(1).strip(),
                        float(mult.group(1)) if mult else 1.0))
    return out


def root_link(urdf: str) -> str:
    """The URDF link that is never a joint's child - the MJCF's root body."""
    links = set(re.findall(r'<link name="([^"]+)"', urdf))
    children = set(re.findall(r'<child link="([^"]+)"', urdf))
    roots = links - children
    if len(roots) != 1:
        sys.exit(f'expected exactly one root link, found {sorted(roots)}')
    return roots.pop()


def head_camera_optics(urdf: str):
    """(width, height, vertical fov in degrees) read from the robot description.

    MuJoCo specifies a camera by its VERTICAL fov, which follows from the
    horizontal one and the aspect ratio. Both head sensors are read and have to
    agree: MuJoCo renders ONE camera for both the colour and the depth stream,
    so optics that differ between them cannot be honoured at all.
    """
    block = re.search(r'<gazebo reference="' + HEAD_CAMERA_LINK + r'">.*?</gazebo>',
                      urdf, re.S)
    if not block:
        sys.exit(f'ERROR: no <gazebo reference="{HEAD_CAMERA_LINK}"> in the URDF, '
                 f'so the head camera has no optics to take')
    specs = set()
    for cam in re.findall(r'<camera [^>]*>.*?</camera>', block.group(0), re.S):
        got = [re.search(tag, cam) for tag in (r'<width>(\d+)</width>',
                                               r'<height>(\d+)</height>',
                                               r'<horizontal_fov>([^<]+)</horizontal_fov>')]
        if not all(got):
            sys.exit('ERROR: the head camera sensor is missing a resolution or '
                     'a field of view; regenerate the URDF with generate_urdf.py')
        w, h, hfov = got
        specs.add((int(w.group(1)), int(h.group(1)), float(hfov.group(1))))
    if len(specs) != 1:
        sys.exit(f'ERROR: the head camera sensors disagree on their optics '
                 f'{sorted(specs)}; MuJoCo has one camera to render both with')
    w, h, hfov = specs.pop()
    return w, h, math.degrees(2 * math.atan(math.tan(hfov / 2) * h / w))


def laser_spec(urdf: str, site: str):
    """(samples, min_angle, max_angle, min_range, max_range, rate) from the description.

    Read from the scanner's own <gazebo> sensor block, so the rays MuJoCo casts
    are the ones the URDF declares rather than a second copy of them.
    """
    block = re.search(r'<gazebo reference="' + site + r'">.*?</gazebo>', urdf, re.S)
    if not block:
        sys.exit(f'ERROR: no <gazebo reference="{site}"> in the URDF, so the '
                 f'scanner on {site} has no specification to take')
    sensor = re.search(r'<sensor [^>]*type="gpu_lidar".*?</sensor>', block.group(0), re.S)
    if not sensor:
        sys.exit(f'ERROR: {site} carries no gpu_lidar sensor to take a scan '
                 f'specification from')
    text = sensor.group(0)
    # Each field is read from the element that owns it: a scan can carry a
    # <vertical> block beside the <horizontal> one, and <range> carries a
    # <min>/<max> pair that has nothing to do with the scan angles.
    rng = re.search(r'<range>.*?</range>', text, re.S)
    hor = re.search(r'<horizontal>.*?</horizontal>', text, re.S)
    fields = {}
    for key, source, pattern in (
            ('samples', hor.group(0) if hor else '', r'<samples>([^<]+)</samples>'),
            ('min_angle', hor.group(0) if hor else '', r'<min_angle>([^<]+)</min_angle>'),
            ('max_angle', hor.group(0) if hor else '', r'<max_angle>([^<]+)</max_angle>'),
            ('min_range', rng.group(0) if rng else '', r'<min>([^<]+)</min>'),
            ('max_range', rng.group(0) if rng else '', r'<max>([^<]+)</max>'),
            ('rate', text, r'<update_rate>([^<]+)</update_rate>')):
        m = re.search(pattern, source)
        if not m:
            sys.exit(f'ERROR: the scanner on {site} declares no {key}; a scan '
                     f'cast from a half-read specification is not the one the '
                     f'description asks for')
        fields[key] = float(m.group(1))
    return (int(fields['samples']), fields['min_angle'], fields['max_angle'],
            fields['min_range'], fields['max_range'], fields['rate'])


def build_mujoco_inputs(urdf: str, spawn_xyz: str, spawn_yaw: str,
                        cam_w: int, cam_h: int, fovy: float,
                        laser_specs: dict) -> str:
    lims = joint_limits(urdf)
    root_body = root_link(urdf)
    actuators = []
    for joint, (kp, clamp, kv) in PAL_GAINS.items():
        if joint not in lims:
            print(f'WARNING: {joint} has no limits in URDF, skipping actuator')
            continue
        lo, hi = lims[joint]
        damping = f'kv="{kv:g}"' if kv is not None else 'dampratio="1.0"'
        actuators.append(
            f'        <position name="{joint}" joint="{joint}" kp="{kp:g}" '
            f'{damping} ctrlrange="{lo:.6g} {hi:.6g}" '
            f'forcerange="{-clamp:g} {clamp:g}"/>')

    # The wheels are commanded joints here, unlike in the competition Gazebo build
    # where the gz mecanum plugin drove them behind ros2_control's back. They are
    # velocity-servoed at PAL's own effort limit, so a stalled base shows up as wheels
    # turning against it -- which is what makes the wheel odometry drift honestly.
    for joint in WHEEL_JOINTS:
        if f'name="{joint}"' not in urdf:
            continue
        actuators.append(
            f'        <velocity name="{joint}" joint="{joint}" kv="{WHEEL_KV:g}" '
            f'forcerange="{-WHEEL_EFFORT:g} {WHEEL_EFFORT:g}"/>')

    # PAL's real arm controllers apply gravity feedforward in firmware, so the
    # PID clamps (26/43 Nm) are headroom on top of gravity, not inclusive of
    # it. Mirror that with MuJoCo gravity compensation on every upper-body
    # link; contact/grasp forces stay real and a carried object is NOT
    # compensated (its weight loads the arm as on the real robot).
    gravcomp = []
    for m in re.finditer(r'<link name="((?:torso_lift|head_[12]|arm_(?:left|right)_\w*|gripper_(?:left|right)\w*)_link)"', urdf):
        gravcomp.append(
            f'        <modify_element type="body" name="{m.group(1)}" gravcomp="1.0"/>')
    gravcomp = sorted(set(gravcomp))


    equalities = []
    superseded = 0
    for mimic, driven, mult in mimic_table(urdf):
        # Both halves of the four-bar go, not just the fingertip: with the
        # outer finger still slaved to the screw by a ratio the loop is
        # over-determined, and the solver back-drives the screw (measured
        # 4.5 mm of tracking error and 5.4% jaw-gap slope error). Dropping
        # both and letting <connect> close the loop reproduces the measured
        # jaw-gap law exactly.
        if re.match(r'gripper_(left|right)_(fingertip|outer_finger)_(left|right)_joint$',
                    mimic):
            superseded += 1
            continue
        equalities.append(
            f'        <joint joint1="{mimic}" joint2="{driven}" '
            f'polycoef="0 {mult:g} 0 0 0" '
            f'solimp="0.95 0.99 0.001" solref="0.005 1"/>')

    wheel_dynamics = '\n'.join(
        f'      <modify_element type="joint" name="{joint}" '
        f'damping="{WHEEL_DAMPING:g}" frictionloss="{WHEEL_FRICTIONLOSS:g}" '
        f'armature="{WHEEL_ARMATURE:g}"/>'
        for joint in WHEEL_JOINTS if f'name="{joint}"' in urdf)

    # The scanners cast their rays on the plugin's worker thread (async), not
    # inside mj_step: 818 rays against the arena's 371k mesh faces cost 5-6 ms
    # per scan, and two scanners at 10 Hz put 220 us on every 2 ms physics
    # step - half of what the whole step costs at rest (measured 2026-09-05,
    # 430 us with the rays in the step against 220 us without). The scan is
    # stamped with the time its rays were cast (plugin_state) and delivered
    # one period later, like a real scanner's sweep.
    lidar_instances, sensors = [], []
    for name, site in LASERS:
        if f'name="{site}"' not in urdf:
            print(f'WARNING: {site} not in URDF, skipping {name}')
            continue
        samples, min_angle, max_angle, min_range, max_range, rate = laser_specs[site]
        lidar_instances.append(
            f'          <instance name="{name}">\n'
            f'            <config key="resolution" value="{samples} 1"/>\n'
            f'            <config key="azimuth_range" '
            f'value="{min_angle:.10g} {max_angle:.10g}"/>\n'
            f'            <config key="elevation_range" value="0.0"/>\n'
            f'            <config key="min_range" value="{min_range:g}"/>\n'
            f'            <config key="max_range" value="{max_range:g}"/>\n'
            f'            <config key="update_rate" value="{rate:g}"/>\n'
            f'            <config key="async" value="1"/>\n'
            f'          </instance>')
        sensors.append(
            f'        <plugin name="{name}" instance="{name}" objtype="site" '
            f'objname="{site}"/>')
    if f'name="{IMU_SITE}"' in urdf:
        sensors.append(
            f'        <framequat name="{IMU_SENSOR}_quat" objtype="site" '
            f'objname="{IMU_SITE}"/>')
        sensors.append(f'        <gyro name="{IMU_SENSOR}_gyro" site="{IMU_SITE}"/>')
        sensors.append(
            f'        <accelerometer name="{IMU_SENSOR}_accel" site="{IMU_SITE}"/>')

    # An <extension> naming mujoco.plugin.lidar makes the engine plugin a hard
    # requirement of the model: if it is not loaded, compilation fails outright
    # with "plugin mujoco.plugin.lidar not found" and the whole simulator dies.
    # So emit it only when a scanner actually made it in.
    lidar_extension = ''
    if lidar_instances:
        lidar_extension = ('      <extension>\n'
                           '        <plugin plugin="mujoco.plugin.lidar">\n'
                           + chr(10).join(lidar_instances) + '\n'
                           '        </plugin>\n'
                           '      </extension>\n')

    connects, excludes = [], []
    for grip in ('left', 'right'):
        for finger in ('left', 'right'):
            tip = f'gripper_{grip}_fingertip_{finger}_link'
            outer = f'gripper_{grip}_outer_finger_{finger}_link'
            inner = f'gripper_{grip}_inner_finger_{finger}_link'
            if f'name="{tip}"' not in urdf:
                continue
            connects.append(
                f'        <connect name="four_bar_{grip}_{finger}" '
                f'body1="{tip}" body2="{outer}" active="true" '
                f'anchor="{FOUR_BAR_ANCHOR[finger]}" '
                f'solimp="0.95 0.99 0.001" solref="0.005 1"/>')
            # The links of a closed loop overlap at their shared pivot, so
            # they must not also collide there.
            excludes.append(f'        <exclude body1="{inner}" body2="{outer}"/>')
            excludes.append(f'        <exclude body1="{tip}" body2="{outer}"/>')
    equalities.extend(connects)
    print(f'{len(actuators)} actuators, {len(equalities)} equalities '
          f'({len(connects)} four-bar loop closures replacing {superseded} '
          f'mimics), {len(excludes)} contact excludes')

    # The robot's D435: depth 87 deg across, colour 69, two sensors (the
    # competition URDF gives one camera the depth's optics for both, which
    # hides any confusion of the two). ERC_DEPTH=exact keeps the URDF's.
    depth_fovy, depth_w, depth_h = fovy, cam_w, cam_h
    if os.environ.get('ERC_DEPTH', 'd435') != 'exact':
        if os.environ.get('ERC_D435_RES') == '1':
            # and the D435's own images, as the robot sends them over its
            # cable (link_bench.sh): 1280x720 colour, 848x480 depth
            cam_w, cam_h, depth_w, depth_h = 1280, 720, 848, 480

        def vfov(hfov, w, h):
            return math.degrees(2 * math.atan(math.tan(math.radians(hfov) / 2) * h / w))
        depth_fovy = vfov(D435_DEPTH_HFOV, depth_w, depth_h)
        fovy = vfov(D435_COLOUR_HFOV, cam_w, cam_h)
    print(f'head camera: colour {cam_w}x{cam_h} fovy {fovy:.1f} deg, '
          f'depth {depth_w}x{depth_h} fovy {depth_fovy:.1f} deg')
    # A camera that is not where TF says, as on the robot at slot 2: ERC_CAM_ERR
    # moves both sensors, ERC_COLOR_ERR the colour on top of it. Six numbers,
    # "dx dy dz rx ry rz": metres along and degrees about the optical frame
    # (x right, y down, z forward), so ry +2 turns the view 2 deg to the right.
    # Commas or spaces between them (restart_sim.sh passes it unquoted: commas).
    # Only the render moves; the URDF and TF keep the nominal frames.
    def cam_err(name):
        v = (os.environ.get(name) or '').replace(',', ' ').split()
        if not v:
            return None
        if len(v) != 6:
            raise SystemExit(f'{name} wants 6 numbers "dx dy dz rx ry rz", got {v}')
        return [float(a) for a in v]
    both, colour_only = cam_err('ERC_CAM_ERR'), cam_err('ERC_COLOR_ERR')
    depth_err = both
    colour_err = ([a + b for a, b in zip(both or [0.0] * 6, colour_only)] if colour_only else both)
    err_attr = lambda e: f' error="{" ".join(repr(a) for a in e)}"' if e else ''
    if both or colour_only:
        print(f'head camera OFF ITS FRAME: depth {depth_err}, colour {colour_err}')

    return f'''  <mujoco_inputs>
    <raw_inputs>
      <!-- multiccd: without it every convex mesh pair yields a single
           contact point - each fingertip touched the book at exactly one
           point (free pivot in the grasp), and the flat book tunnelled
           10 mm into the bin's thin coacd floor sheets (measured
           2026-08-26).
           noslip: the soft-contact solver lets a loaded contact drift along
           its friction directions. A book pinched at 10 N per pad with its
           centre 37 mm behind the pinch axis turned in the pads at about a
           degree a second and fell out after 70 s; a real book does not
           turn in a hand that holds it. With the noslip post-processor the
           same hold drifts 0.3 deg in four minutes (measured 2026-09-05).
           100 and not 5: 5 holds a still hand but lets the book creep while
           the arm moves; turn in the pads at the release 14.5 -> 7.1 deg
           median (max 19.6 -> 11.5), 0 of 15 over 15 deg (was 7 of 15), the
           bin not moved, 1.11x the wall time (tiger #65, 2026-11-02).
           sleep: a free body that has been still for 10 steps stops being
           simulated until something awake touches it or its qpos is set.
           The 20 resting books were 80 of the arena's 88 contacts and most
           of the constraint solve; asleep they cost nothing, and the robot
           (actuated, so never asleep) wakes whatever it touches. mj_step
           719 us -> 286 us on the full arena (measured 2026-09-05). -->
      <option integrator="implicitfast" cone="elliptic" impratio="10" noslip_iterations="100">
        <flag multiccd="enable" sleep="enable"/>
      </option>
      <!-- znear is a fraction of the scene's statistic extent (2 m), so 0.005
           puts the near plane at 1 cm: a colour camera has no minimum range,
           and at the old 0.2 m an arm passing the lens was clipped out of a
           head-camera picture the real one would have been filled by (issue
           #59). The camera does not see its own housing at 1 cm (rendered).
           The depth stream's minimum range is the camera plugin's min_depth. -->
      <visual>
        <map znear="0.005" zfar="30"/>
      </visual>
      <default>
        <default class="visual">
          <geom group="2" type="mesh" contype="0" conaffinity="0"/>
        </default>
        <default class="collision">
          <!-- contype 1 / conaffinity 2 disables robot self-collision while
               keeping robot-vs-world contact (world geoms are 1/1): the
               arms' zero pose otherwise presses into the chassis and gets
               pushed out to a dangling pose that fouls the table. -->
          <geom group="3" type="mesh" contype="1" conaffinity="2"/>
        </default>
      </default>
      <actuator>
{chr(10).join(actuators)}
      </actuator>
{lidar_extension}      <sensor>
{chr(10).join(sensors)}
      </sensor>
      <equality>
{chr(10).join(equalities)}
      </equality>
      <contact>
{chr(10).join(excludes)}
      </contact>
    </raw_inputs>
    <processed_inputs>
      <!-- The converter treats the site as a REP-103 optical frame and applies
           the optical->MuJoCo rotation itself. -->
      <camera site="head_front_camera_color_optical_frame" name="head_front_camera"
              fovy="{fovy:.6g}" mode="fixed" resolution="{cam_w} {cam_h}"{err_attr(colour_err)}/>
      <!-- The depth sensor is a camera of its own, at its own optical frame
           and with its own field of view: on the robot's D435 the depth sees
           87 deg across where the colour sees 69, from 15 mm to the side, and
           a stack that takes one image's pixel for the other's ray is wrong
           there (slot 1, 2026-10-31). ERC_DEPTH=exact gives it the colour's. -->
      <camera site="head_front_camera_depth_optical_frame" name="head_front_camera_depth"
              fovy="{depth_fovy:.6g}" mode="fixed" resolution="{depth_w} {depth_h}"{err_attr(depth_err)}/>
      <!-- The mecanum roller pattern is not modelled as geometry: MuJoCo has no
           equivalent of the fdir1 the competition's Gazebo build uses to point each
           wheel's friction along its 45-degree roller axis, and only a capsule's
           anisotropic friction frame follows the geom, which a spinning wheel cannot
           exploit. BaseVelocityPlugin's traction mode supplies that force analytically
           from the wheels' measured rotation instead, so the wheel geoms are left
           near-frictionless: they carry the robot's weight and collide with the world,
           but they must not also fight the base servo. Geoms are addressed by
           (mesh, class). -->
      <!-- priority=1 makes the wheel's friction win outright over the floor's
           (default combination is element-wise max, so lowering only the
           wheel would do nothing). -->
      <modify_element type="geom" mesh="wheel_link" class="collision" friction="{WHEEL_FRICTION}" priority="1"/>
      <modify_element type="geom" mesh="wheel_link_reflected" class="collision" friction="{WHEEL_FRICTION}" priority="1"/>
{wheel_dynamics}
      <!-- The pad (PAD_MU, priority 3, PAD_SOLREF: see there); condim 6 gives the pinch the
           torsional/rolling friction that resists the book pivoting about the
           grasp axis (the higher priority's condim governs the pair). -->
      <modify_element type="geom" mesh="fingertip" class="collision" friction="{PAD_MU:g} 0.015 0.002" condim="6" priority="3" solref="{PAD_SOLREF:g} 1"/>
{chr(10).join(gravcomp)}
      <!-- Where the robot starts. The root body carries the free joint, so its
           pos/quat are the floating base's initial qpos; the arena spec starts
           the robot yawed 90 degrees on the start zone. Spawn it on the floor:
           the base is carried by its wheels' contact with the ground, so a
           robot spawned in the air falls until they reach it. -->
      <modify_element type="body" name="{root_body}" pos="{spawn_xyz}" euler="0 0 {spawn_yaw}"/>
    </processed_inputs>
  </mujoco_inputs>
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--headless', default='true', choices=['true', 'false'])
    ap.add_argument('--camera-rate', default='15.0')
    ap.add_argument('--sim-speed-factor', default='-1.0',
                    help='-1 follows the viewer slowdown setting')
    ap.add_argument('--spawn-xyz', default='0 0 0',
                    help='floating base start position, MJCF order')
    ap.add_argument('--spawn-yaw', default='0',
                    help='floating base start yaw in radians')
    args = ap.parse_args()

    src = get_package_share_directory('erc_description') + '/urdf/tiago_pro.urdf'
    urdf = open(src).read()

    cam_w, cam_h, fovy = head_camera_optics(urdf)
    # Both readers have to run before strip_gazebo_blocks deletes their source.
    laser_specs = {site: laser_spec(urdf, site)
                   for _, site in LASERS if f'name="{site}"' in urdf}

    urdf = strip_gazebo_blocks(urdf)
    urdf = strip_transmissions(urdf)
    urdf = strip_laser_housings(urdf)
    urdf = command_wheels(urdf)

    hardware = f'''<plugin>mujoco_ros2_control/MujocoSystemInterface</plugin>
      <param name="mujoco_model_topic">/mujoco_robot_description</param>
      <param name="headless">{args.headless}</param>
      <param name="camera_publish_rate">{args.camera_rate}</param>
      <param name="sim_speed_factor">{args.sim_speed_factor}</param>'''
    if not HARDWARE_PLUGIN_RE.search(urdf):
        sys.exit('ERROR: no simulator hardware plugin found in <ros2_control>, '
                 'so there is nothing to swap for the MuJoCo one')
    urdf = HARDWARE_PLUGIN_RE.sub(lambda _m: hardware, urdf, count=1)

    imu_iface = ''
    if f'name="{IMU_SITE}"' in urdf:
        interfaces = ''.join(
            f'      <state_interface name="{n}"/>\n' for n in (
                'orientation.x', 'orientation.y', 'orientation.z', 'orientation.w',
                'angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z',
                'linear_acceleration.x', 'linear_acceleration.y',
                'linear_acceleration.z'))
        imu_iface = (f'    <sensor name="{IMU_SENSOR}">\n'
                     f'      <param name="mujoco_type">imu</param>\n'
                     f'{interfaces}'
                     f'    </sensor>\n')
        urdf = urdf.replace('</ros2_control>', imu_iface + '  </ros2_control>', 1)

    inputs = build_mujoco_inputs(urdf, args.spawn_xyz, args.spawn_yaw,
                                 cam_w, cam_h, fovy, laser_specs)
    urdf = urdf.replace('</robot>', inputs + '</robot>', 1)

    with open(args.output, 'w') as f:
        f.write(urdf)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
