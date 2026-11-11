#!/usr/bin/env python3
"""
Generate the competition arena as a MuJoCo scene (MJCF).

This file IS the arena definition: floor, walls, start zone, table, shelf,
collection bin, books and number markers. Poses, sizes, masses and frictions
below are the competition's own, carried over from the SDF world and model
files the arena was previously described by. The MJCF converter inlines this
file's children into the robot model, so the robot and the arena end up in one
MuJoCo model.

Two things force this to be generated rather than checked in:

  * mesh and texture paths must be absolute (the converter copies this file's
    contents into an MJCF written somewhere else, so relative paths would
    resolve against the wrong directory), and
  * the book colour layout and the number-marker order are drawn from ERC_SEED
    at launch.

Collision geometry is not the visual mesh: MuJoCo collides a mesh as its convex
hull, which would make the shelf a solid slab with nowhere to put a book and
fill the leg room under the table. The table and shelf are decomposed into
exact boxes by mesh_collision.py; the bin is bevelled rather than slab-shaped,
so its interior is described by hand-fitted primitives measured from the mesh.

    ros2 run erc_bringup generate_mujoco_world.py -o /tmp/erc_world.xml
    ERC_SEED=7 ros2 run erc_bringup generate_mujoco_world.py -o /tmp/erc_world.xml
"""

import argparse
import math
import os
import random
import sys

from ament_index_python.packages import get_package_share_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh_collision import decompose  # noqa: E402

# ── Arena layout ────────────────────────────────────────────────────────────

BOOK_COLOURS = {
    'red':    (1.0, 0.0, 0.0, 1.0),
    'green':  (0.0, 1.0, 0.0, 1.0),
    'yellow': (1.0, 1.0, 0.0, 1.0),
    'blue':   (0.0, 0.0, 1.0, 1.0),
}

SHELF_X, SHELF_Y, SHELF_Z = 3.0, 0.0, 1.1
NUM_COLUMNS, NUM_ROWS = 5, 6
ACTIVE_ROWS = [1, 2, 3, 4]
COLUMN_WIDTH = 1.0
COLUMN_JITTER_RANGE = COLUMN_WIDTH * 0.25
COLUMN_Y_OFFSETS = [((NUM_COLUMNS - 1) / 2 - c) * COLUMN_WIDTH for c in range(NUM_COLUMNS)]
TOP_BOOK_Z = 0.825
ROW_SPACING = 0.33
ROW_FLOOR_Z_OFFSETS = [TOP_BOOK_Z - (i * ROW_SPACING) for i in range(NUM_ROWS)]
NUMBER_MARKER_PLATE_X = SHELF_X - 0.245
NUMBER_MARKER_PLATE_Z = 1.93         # phase 2: on the top shelf row, in reach of the head camera (upstream cc3f2e0)

# Round 2 (upstream simulation_round_2.launch.py): columns 1, 3, 5 as round 1;
# column 2 holds four white books whose spines carry a word, shuffled by the
# seed; column 4 holds coloured books flush against the column's side walls,
# alternating left/right from the top row down. Round 2 lays books out on a
# 1.03 m pitch where round 1 uses 1.0; both are kept so a seed reproduces the
# organisers' layout draw for draw.
ROUND2_PITCH = COLUMN_WIDTH + 0.03
WHITE_COLUMN, EDGE_COLUMN = 2, 4
WHITE_BOOKS = [('A', 'A1'), ('B', 'B1'), ('C', 'C1'), ('D', 'D1')]   # (cover, spine) textures
# The words the spine textures carry, for a judge; the robot has to read them.
WHITE_WORDS = {'A': 'FUTURE', 'B': 'ROBOT', 'C': 'LIBRARY', 'D': 'AGENT'}
# The shelf mesh's dividers (mesh_collision.py's exact boxes): 35 mm thick on a
# 1.035 m pitch, so column c's inner faces sit at its centre +-0.5. Upstream
# puts its edge books at the 1.03 grid +-0.485, which leaves one of each pair
# 5 mm inside the divider and the other 5 mm clear of it; the physical shelf
# has them flush, so this does: book face 0.5 mm off the divider's face.
SHELF_DIVIDER_PITCH = 1.035
EDGE_CLEARANCE = 0.0005

BOOK_HALF = (0.125, 0.015, 0.08)     # the competition's 0.25 x 0.03 x 0.16 box (upstream cc3f2e0)
BOOK_MASS = 0.3
# Upstream sets mu 10 so books never slip in Gazebo. A grip or a drag that only
# works at mu 10 fails on the real robot, so the dev sim uses realistic values
# and sweeps them: this is a book's paper cover on a wooden board (ERC_BOOK_MU).
# priority="2" makes it govern the book's contacts with the shelf, the bin and
# other books (MuJoCo otherwise takes the larger of a pair), condim included;
# the fingertip pads carry their own friction and condim at a higher priority,
# 3 (generate_mujoco_urdf.py PAD_MU), so a pinch and a drag each see the two
# frictions a real one would.
BOOK_MU = 0.5
MARKER_HALF = (0.15, 0.15, 0.01)     # the competition's 0.3 x 0.3 plate face

# Every arena prop is placed with the same pose rpy "1.5708 0 -1.5708",
# which is this quaternion. Under it mesh +X is world -Y, mesh +Y is world +Z
# and mesh +Z is world -X.
PROP_QUAT = '0.5 0.5 -0.5 -0.5'

# Pitch the book 90 degrees about Y so it stands on the shelf like a book.
BOOK_QUAT = '0.70710678 0 0.70710678 0'
# The number plates are plane geoms, not boxes: MuJoCo maps a 2d texture across
# a plane but only samples a single line of it across a box face, which renders
# the digits as a blank white slab. These axes put the plane's normal along -X
# (facing the robot) with the digit upright and reading the right way round.
MARKER_XYAXES = '0 -1 0 0 0 1'
# The white books' printed faces. A plane is only allowed on a static body, so
# each print is a single textured quad (a flat mesh, shell inertia, no mass) a
# hair off the box face, in book-local axes: the body's x is world -z, its y
# world y, its z world +x. The spine, at local z -half, faces the robot with
# the image's columns across it (u along world -y) and its rows top to bottom
# (v along world -z), as upstream paints it; the covers sit on the two large
# +-y faces with the image's long side along the book's height.
PRINT_PROUD = 0.0005


def _quad(name, verts, uv, faces):
    return (f'    <mesh name="{name}" inertia="shell" vertex="{fmt(v for p in verts for v in p)}" '
            f'texcoord="{fmt(t for p in uv for t in p)}" face="{faces}"/>')


def print_meshes():
    hx, hy, hz = BOOK_HALF
    z = -(hz + PRINT_PROUD)
    yp = hy + PRINT_PROUD
    corners = ((0, 0), (1, 0), (1, 1), (0, 1))
    return [
        # spine: top-left, top-right, bottom-right, bottom-left as seen from the robot
        _quad('white_spine', [(-hx, hy, z), (-hx, -hy, z), (hx, -hy, z), (hx, hy, z)],
              corners, '0 2 1 0 3 2'),
        _quad('white_cover_p', [(-hx, yp, -hz), (hx, yp, -hz), (hx, yp, hz), (-hx, yp, hz)],
              corners, '0 2 1 0 3 2'),
        _quad('white_cover_n', [(-hx, -yp, -hz), (hx, -yp, -hz), (hx, -yp, hz), (-hx, -yp, hz)],
              corners, '0 1 2 0 2 3'),
    ]

# The collection bin's interior, in bin-local coordinates. The mesh is bevelled,
# so mesh_collision.py refuses it and these are measured from the mesh instead:
# the interior floor's top face is at bin-local y -0.095, which is world z
# 0.750 once the bin rests on the table; 0.31 x 0.56 clear, walls 0.20 tall,
# and a lower lip at the open front edge. Sizes below are MuJoCo half-extents.
# Named so a contact check can tell the floor from the walls.
BIN_COLLISION = [
    ('bin_floor',      (0.0, -0.100, -0.025), (0.155, 0.005, 0.280)),
    ('bin_wall_back',  (0.0, 0.005, 0.2498),  (0.155, 0.100, 0.0052)),
    ('bin_wall_left',  (-0.15, 0.005, -0.025), (0.005, 0.100, 0.280)),
    ('bin_wall_right', (0.15, 0.005, -0.025), (0.005, 0.100, 0.280)),
    ('bin_lip_front',  (0.0, -0.055, -0.2927), (0.155, 0.040, 0.0122)),
]
BIN_MASS = 4.70313898335195
# Straight from the competition's <inertial>: the centre of mass sits 34.9 mm off the
# body origin, and ixy is 7% of ixx, so neither can be dropped without changing
# how the bin rocks when a book lands in it.
BIN_COM = '-0.000869 -0.034922 0'
BIN_INERTIA = '0.0896407437951023 0.223231929617035 0.175200579933907 0.00627075393219412 0 0'
# Spawned already resting on the table. The arena spec drops it from z 1.3 onto
# a table whose top is at 0.74; it settles here, and starting it settled keeps
# the arena deterministic instead of depending on how a 4.7 kg box bounces.
BIN_POSE = (-1.0, 0.0, 0.845)
TABLE_POSE = (-1.0, 0.0, 0.7)

# A drawn layout (--layout N, $ERC_LAYOUT): at the venue the table's distance
# from the shelf and the robot's start pose are randomised by an amount nobody
# has published. The table moves with the bin on it, along and across the arena
# and turned about its centre; the robot starts with its whole base inside the
# painted start zone (start_fit), as it must at the competition, whose zone is
# a square on the floor. These ranges are a guess to be widened once one is known.
LAYOUT_TABLE_X = (-1.6, -1.1)        # table centre; nearer, a turned corner reaches a turning robot
LAYOUT_TABLE_Y = (-0.6, 0.6)
LAYOUT_TABLE_YAW = 0.35              # +- rad, about 20 degrees
LAYOUT_START_XY = 0.1                # +- m from the start zone's centre, drawn
START_YAW = 1.5708                   # the arena spec's start heading
START_ZONE_HALF = 0.4                # m: the painted zone, 0.8 m square, axis-aligned
# The TIAGo Pro base's half extents, metres (the mission's tiger/footprint.py
# BASE_HALF and robot.BASE_HALF; 0.72 x 0.49 m): stated here because the
# simulator does not import the mission's package.
BASE_HALF = (0.36, 0.245)
# The base 0.72 m long in a 0.8 m zone has 40 mm to spare along its length at
# the zone's own heading and centre, and every degree turned costs 4.3 mm of
# it. The start keeps START_SPARE of those 40 mm: it is where the mission
# drives home to, so it is the homing's whole budget, and 20 mm is what that
# took on the bench (the creep's 10 mm tolerance plus the 10-21 mm the fix at
# home corrects, #68). The other 20 mm is shared between a turn off the axis,
# up to START_TURN, and an offset along the base's length.
START_SPARE = 0.020
START_TURN = math.radians(3.0)


# The robot spawns with its arms in the load posture, out 0.77 m ahead of the
# base at 0.36 m up: a start facing a table nearer than this has them in its top
# (a drawn start turned to face the table, layout 5: 31 mm into it from the spawn).
START_FACING_RANGE = 2.0


def start_fit(x, y, yaw, table=None):
    """A drawn start (offsets up to LAYOUT_START_XY, any heading) brought inside
    the zone: the heading to the nearest axis that does not face `table` (its
    centre x, y) from within START_FACING_RANGE, plus up to START_TURN in the
    same proportion the draw was off its nearest axis, and each offset scaled
    down to what the base leaves at that heading with START_SPARE kept. The same
    three numbers in, so the layout's other draws are unchanged."""
    def facing(axis):
        if table is None:
            return False
        dx, dy = table[0] - x, table[1] - y
        off = math.atan2(math.sin(math.atan2(dy, dx) - axis), math.cos(math.atan2(dy, dx) - axis))
        return math.hypot(dx, dy) < START_FACING_RANGE and abs(off) < math.pi / 4
    nearest = round(yaw / (math.pi / 2)) * (math.pi / 2)
    turn = (yaw - nearest) / (math.pi / 4) * START_TURN
    axes = sorted((nearest + k * math.pi / 2 for k in range(4)),
                  key=lambda a: abs(math.atan2(math.sin(yaw - a), math.cos(yaw - a))))
    axis = next((a for a in axes if not facing(a)), nearest)
    c, s = abs(math.cos(turn)), abs(math.sin(turn))
    along = START_ZONE_HALF - START_SPARE - (BASE_HALF[0] * c + BASE_HALF[1] * s)
    across = START_ZONE_HALF - START_SPARE - (BASE_HALF[0] * s + BASE_HALF[1] * c)
    if round(axis / (math.pi / 2)) % 2:          # the base's length along the zone's y
        along, across = across, along
    return (x * min(along, LAYOUT_START_XY) / LAYOUT_START_XY,
            y * min(across, LAYOUT_START_XY) / LAYOUT_START_XY,
            math.atan2(math.sin(axis + turn), math.cos(axis + turn)))


def draw_layout(n):
    """(table x, y, yaw, start x, y, yaw) for layout `n`, or the arena spec's
    fixed one for None. Its own generator, so the books' draw from ERC_SEED is
    the same whatever the layout."""
    if n in (None, ''):
        return TABLE_POSE[0], TABLE_POSE[1], 0.0, 0.0, 0.0, START_YAW
    g = random.Random(int(n))
    if os.environ.get('ERC_LAYOUT_MODE') == 'anywhere':
        # The robot anywhere on the open floor - near the shelf, across the
        # room, beside the wall - at a heading near an axis, and the table
        # anywhere round it. The start zone is painted where it starts.
        sx, sy = g.uniform(-3.2, 1.3), g.uniform(-3.6, 3.6)
        while True:
            # the table's corner (0.75 m off its centre) at least 0.5 m clear
            # of the robot (0.4 m round): near, but not touching
            a, r = g.uniform(-math.pi, math.pi), g.uniform(1.65, 2.3)
            tx, ty = sx + r * math.cos(a), sy + r * math.sin(a)
            if tx < 1.0 and -3.4 < tx and abs(ty) < 3.9:
                break
        # (the zone is painted round the start, so only the heading is fitted)
        tyaw = g.uniform(-math.pi, math.pi)
        return tx, ty, tyaw, sx, sy, start_fit(0.0, 0.0, g.uniform(-math.pi, math.pi), (tx - sx, ty - sy))[2]
    if os.environ.get('ERC_LAYOUT_MODE') == 'ring':
        # The robot may start on any side of the table, near it or far: the
        # table anywhere round the start - beside it, behind it, between it and
        # the shelf - from just clear of the robot out to 2 m, turned any way,
        # and never on the ground in front of the shelf the robot must stand on.
        while True:
            a, r = g.uniform(-math.pi, math.pi), g.uniform(1.15, 2.0)
            tx, ty = r * math.cos(a), r * math.sin(a)
            if tx < 1.0 and -3.2 < tx and abs(ty) < 3.8:
                break
        tyaw = g.uniform(-math.pi, math.pi)
        return (tx, ty, tyaw, *start_fit(g.uniform(-LAYOUT_START_XY, LAYOUT_START_XY),
                                         g.uniform(-LAYOUT_START_XY, LAYOUT_START_XY),
                                         g.uniform(-math.pi, math.pi), (tx, ty)))
    tx, ty, tyaw = (g.uniform(*LAYOUT_TABLE_X), g.uniform(*LAYOUT_TABLE_Y),
                    g.uniform(-LAYOUT_TABLE_YAW, LAYOUT_TABLE_YAW))
    return (tx, ty, tyaw, *start_fit(g.uniform(-LAYOUT_START_XY, LAYOUT_START_XY),
                                     g.uniform(-LAYOUT_START_XY, LAYOUT_START_XY),
                                     g.uniform(-math.pi, math.pi), (tx, ty)))


# ── The venue (ERC_VENUE=mbr): the MBR library's wall, from photos of a team's
#    slot (2026-10-30). The shelf is the bottom-right three bays x five rows of
#    a white library wall whose other bays are packed with real books - to the
#    left and above, or on both sides (ERC_VENUE_WALL=both: a section in the
#    middle of the wall) - on a white plinth over a glossy white floor. A grey
#    card with a dark digit hangs from the top board in the top row of a bay
#    (inverted: white on black, as it photographed under one exposure). Only
#    some bays may be set up (ERC_VENUE_BAYS) and only some labelled
#    (ERC_VENUE_LABELS "bay:digit,..."). The rows keep the arena's book heights,
#    so the stack meets the same reach; the bays keep the mesh's 1.035 m pitch.
#    ERC_CLUTTER=N adds N other tables and ledges round the arena, away from
#    the paths; ERC_OPEN=1 takes the arena's wall away. ──
VENUE_FACE_X = 2.75
VENUE_DEPTH = 0.32
VENUE_BAY = 1.035
VENUE_DIV_T = 0.035
VENUE_BOARD_T = 0.025
VENUE_PLINTH = 0.455                       # the bottom board's top at 0.48
VENUE_TOPS = [0.48 + 0.33 * i for i in range(12)]     # 0.48 .. 4.11: the section's six, the wall's above
VENUE_SECTION_ROWS = 6                     # boards 0.48 .. 2.13: four rows of books, the plate row
VENUE_PALETTE = [(0.55, 0.74, 0.88), (0.62, 0.80, 0.92), (0.90, 0.91, 0.92), (0.78, 0.80, 0.82),
                 (0.55, 0.57, 0.60), (0.30, 0.32, 0.35), (0.08, 0.08, 0.09), (0.16, 0.26, 0.45),
                 (0.88, 0.85, 0.76), (0.40, 0.58, 0.72)]


def venue_bays(wall):
    """(y, kind) of every bay of the wall: 'section' 1..3 left to right as the
    robot faces it, then the packed ones."""
    out = [((2 - k) * VENUE_BAY, ('section', k)) for k in (1, 2, 3)]
    # three packed bays a side keep the wall inside the arena's +-5 m
    out += [((1 + k) * VENUE_BAY, ('wall', None)) for k in range(1, 4)]
    if wall == 'both':
        out += [(-(1 + k) * VENUE_BAY, ('wall', None)) for k in range(1, 4)]
    return out


def venue_shelf(wall, rng):
    """The wall's frame (boxes on the erc_shelf body, which arena.py reads as
    the shelf) and its packed books (visual, plus one collision block per
    packed bay-row)."""
    bays = venue_bays(wall)
    ys = [y for y, _ in bays]
    lo, hi = min(ys) - VENUE_BAY / 2, max(ys) + VENUE_BAY / 2
    cx = VENUE_FACE_X + VENUE_DEPTH / 2
    top = VENUE_TOPS[-1]
    # group 2 as every other arena visual: the robot's head camera renders that
    # group, and the frame was invisible to it while the books in it showed
    # (the venue's world, 2026-10-30); collision is unchanged
    white = 'group="2" rgba="0.93 0.93 0.92 1"'
    g = []

    def box(name, pos, half, attrs=white):
        g.append(f'      <geom name="{name}" type="box" pos="{fmt(pos)}" size="{fmt(half)}" {attrs}/>')
    box('venue_plinth', (cx, (lo + hi) / 2, VENUE_PLINTH / 2), (VENUE_DEPTH / 2, (hi - lo) / 2, VENUE_PLINTH / 2))
    box('venue_back', (VENUE_FACE_X + VENUE_DEPTH - 0.01, (lo + hi) / 2, (VENUE_PLINTH + top) / 2),
        (0.01, (hi - lo) / 2, (top - VENUE_PLINTH) / 2))
    edges = sorted({round(y + d * VENUE_BAY / 2, 4) for y in ys for d in (-1, 1)})
    for i, y in enumerate(edges):
        box(f'venue_divider_{i}', (cx, y, (VENUE_PLINTH + top) / 2),
            (VENUE_DEPTH / 2, VENUE_DIV_T / 2, (top - VENUE_PLINTH) / 2))
    # each board one plank the wall's width, as the arena mesh's are (the
    # dividers stand through them): arena.board_under looks for exactly that
    for r, t in enumerate(VENUE_TOPS):
        box(f'venue_board_{r}', (cx, (lo + hi) / 2, t - VENUE_BOARD_T / 2),
            (VENUE_DEPTH / 2, (hi - lo) / 2, VENUE_BOARD_T / 2))
    lib = []
    for b, (y, (kind, k)) in enumerate(bays):
        for r, t in enumerate(VENUE_TOPS):
            if r + 1 >= len(VENUE_TOPS):
                continue
            if kind == 'section' and r < VENUE_SECTION_ROWS - 1:
                continue
            # packed: books from divider to divider, spines flush with the front
            ceil = VENUE_TOPS[r + 1] - VENUE_BOARD_T - t
            yy = y + VENUE_BAY / 2 - VENUE_DIV_T / 2 - 0.004
            end = y - VENUE_BAY / 2 + VENUE_DIV_T / 2 + 0.004
            n = 0
            while True:
                w = rng.uniform(0.016, 0.05)
                if yy - w < end:
                    break
                if rng.random() < 0.04:          # a gap now and then
                    yy -= w
                    continue
                hgt = min(ceil - 0.01, rng.uniform(0.17, 0.30))
                d = rng.uniform(0.15, 0.24)
                c = rng.choice(VENUE_PALETTE)
                shade = rng.uniform(0.9, 1.05)
                rgba = tuple(min(1.0, v * shade) for v in c) + (1.0,)
                lib.append(f'      <geom name="lib_{b}_{r}_{n}" class="arena_visual" type="box" '
                           f'pos="{fmt((VENUE_FACE_X + 0.005 + rng.uniform(0, 0.012) + d / 2, yy - w / 2, t + hgt / 2))}" '
                           f'size="{fmt((d / 2, w / 2 - 0.0005, hgt / 2))}" rgba="{fmt(rgba)}"/>')
                yy -= w
                n += 1
            # what a wrong turn would hit: one block for the row's books
            box(f'venue_packed_{b}_{r}', (VENUE_FACE_X + 0.015 + 0.1, y, t + 0.12),
                (0.1, VENUE_BAY / 2 - VENUE_DIV_T, 0.12), 'class="arena_collision"')
    return g, lib


def venue_markers(labels, style, marker_tex_dir):
    """Assets and geoms of the bays' labels: {bay: digit}."""
    assets, geoms = [], []
    for bay, digit in sorted(labels.items()):
        tex = os.path.join(marker_tex_dir, f'{digit}.png')
        if style == 'inverted':
            src = tex
            tex = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'erc_marker_inv_{digit}.png')
            try:
                import cv2
                cv2.imwrite(tex, 255 - cv2.imread(src))
            except ImportError:
                from PIL import Image, ImageOps
                ImageOps.invert(Image.open(src).convert('RGB')).save(tex)
            assets.append(f'    <texture type="2d" name="marker_inv_{digit}" file="{tex}"/>')
            assets.append(f'    <material name="marker_inv_{digit}" texture="marker_inv_{digit}" '
                          f'texrepeat="1 1" texuniform="false"/>')
            mat = f'marker_inv_{digit}'
        else:
            # a grey card, a dark digit: the plate as it photographed close up
            assets.append(f'    <texture type="2d" name="marker_grey_{digit}" file="{tex}"/>')
            assets.append(f'    <material name="marker_grey_{digit}" texture="marker_grey_{digit}" '
                          f'rgba="0.72 0.72 0.72 1" texrepeat="1 1" texuniform="false"/>')
            mat = f'marker_grey_{digit}'
        y = (2 - bay) * VENUE_BAY
        z = VENUE_TOPS[VENUE_SECTION_ROWS - 1] - VENUE_BOARD_T - MARKER_HALF[0] - 0.02
        geoms.append(f'    <geom name="number_marker_col_{bay}" class="arena_visual" type="plane" '
                     f'pos="{VENUE_FACE_X + 0.03:g} {y:g} {z:g}" xyaxes="{MARKER_XYAXES}" '
                     f'size="{fmt(MARKER_HALF)}" material="{mat}"/>')
    return assets, geoms


def venue_books(bays, book_mu, white=None):
    """One book per row 1-4 of each set-up bay, colours shuffled per bay, a y
    jitter each, at the arena's own heights (the rows' boards are at them).
    `white` (bay, cover): a white book with its word on cover and spine (round
    2's), standing beside the coloured book of the bay's second row - the
    venue's close-up photo has "Future" beside the green one."""
    body = []
    for bay in bays:
        colours = list(BOOK_COLOURS.keys())
        random.shuffle(colours)
        for i, row in enumerate(ACTIVE_ROWS):
            y = (2 - bay) * VENUE_BAY + random.uniform(-COLUMN_JITTER_RANGE, COLUMN_JITTER_RANGE)
            z = SHELF_Z + ROW_FLOOR_Z_OFFSETS[row]
            name = f'book_col_{bay}_row_{row + 1}_{colours[i]}'
            body += book_body(name, (VENUE_FACE_X + 0.15, y, z + 0.002), f'book_{colours[i]}', book_mu)
            if white and white[0] == bay and i == 1:
                # on the other side of the bay from its neighbour, 0.3 m clear
                # of it; or, given a gap, that far from it toward the bay's
                # middle - two kinds of book side by side in one row
                c = (2 - bay) * VENUE_BAY
                if len(white) > 2:
                    wy = y - white[2] if y > c else y + white[2]
                else:
                    wy = c - 0.3 if y > c else c + 0.3
                cover = white[1]
                spine = dict(WHITE_BOOKS)[cover]
                wname = f'book_col_{bay}_row_{row + 1}_white_{cover}'
                body += book_body(wname, (VENUE_FACE_X + 0.15, wy, z + 0.002), 'book_white', book_mu,
                                  white_prints(wname, cover, spine))
    return body


def clutter(n, rng, start=(0.0, 0.0), table=(TABLE_POSE[0], TABLE_POSE[1])):
    """N other tables and ledges round the arena, kept off the start, the
    table, and the ground in front of the shelf the robot must stand on."""
    out, placed = [], []
    tries = 0
    while len(placed) < n and tries < 2000:
        tries += 1
        x, y = rng.uniform(-3.7, 2.3), rng.uniform(-4.6, 4.6)
        if x > 1.3 and abs(y) < 2.4:
            continue
        if math.hypot(x - start[0], y - start[1]) < 1.5 or math.hypot(x - table[0], y - table[1]) < 1.8:
            continue
        if any(math.hypot(x - px, y - py) < 1.3 for px, py in placed):
            continue
        yaw = rng.uniform(-math.pi, math.pi)
        q = _yawed('1 0 0 0', yaw)
        i = len(placed)
        placed.append((x, y))
        if rng.random() < 0.5:
            hx, hy, h = rng.uniform(0.35, 0.6), rng.uniform(0.3, 0.45), rng.uniform(0.70, 0.78)
            out.append(f'    <body name="clutter_table_{i}" pos="{x:g} {y:g} 0" quat="{q}">')
            out.append(f'      <geom type="box" pos="0 0 {h - 0.015:g}" size="{hx:g} {hy:g} 0.015" rgba="0.85 0.85 0.83 1"/>')
            for sx in (-1, 1):
                for sy in (-1, 1):
                    out.append(f'      <geom type="box" pos="{sx * (hx - 0.03):g} {sy * (hy - 0.03):g} {(h - 0.03) / 2:g}" '
                               f'size="0.02 0.02 {(h - 0.03) / 2:g}" rgba="0.6 0.6 0.6 1"/>')
            out.append('    </body>')
        else:
            hx, hy, hz = rng.uniform(0.4, 1.0), rng.uniform(0.05, 0.2), rng.uniform(0.04, 0.35)
            out.append(f'    <geom name="clutter_ledge_{i}" type="box" pos="{x:g} {y:g} {hz:g}" quat="{q}" '
                       f'size="{hx:g} {hy:g} {hz:g}" rgba="0.75 0.74 0.72 1"/>')
    return out


def _yawed(quat, yaw):
    """`quat` (w x y z, as a string) turned by `yaw` about world Z."""
    w1, x1, y1, z1 = math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)
    w2, x2, y2, z2 = (float(v) for v in quat.split())
    q = (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)
    return ' '.join(f'{v:.7g}' for v in q)


def fmt(values):
    return ' '.join(f'{v:g}' for v in values)


def collision_geoms(mesh_path, name, friction=None, indent=6):
    """Exact box decomposition of a slab-shaped mesh, as MJCF geoms."""
    pad = ' ' * indent
    extra = f' friction="{friction}"' if friction else ''
    out = []
    for i, (centre, half) in enumerate(decompose(mesh_path)):
        out.append(f'{pad}<geom name="{name}_collision_{i}" class="arena_collision" '
                   f'type="box" pos="{fmt(centre)}" size="{fmt(half)}"{extra}/>')
    return '\n'.join(out)


def visual_mesh(name, mesh_name, rgba, indent=6):
    pad = ' ' * indent
    return (f'{pad}<geom name="{name}_visual" class="arena_visual" '
            f'type="mesh" mesh="{mesh_name}" rgba="{fmt(rgba)}"/>')


def book_friction(mu):
    return f'{mu:g} 0.01 0.002'


def book_body(name, pos, material, mu, prints=()):
    """A free standing book; `prints` are extra textured planes (white books)."""
    x, y, z = pos
    out = [f'    <body name="{name}" pos="{x:g} {y:.6g} {z:g}" quat="{BOOK_QUAT}">',
           f'      <freejoint name="{name}_joint"/>',
           f'      <geom name="{name}_geom" type="box" size="{fmt(BOOK_HALF)}" '
           f'mass="{BOOK_MASS:g}" friction="{book_friction(mu)}" condim="6" priority="2" '
           f'material="{material}"/>']
    out += [f'      {p}' for p in prints]
    out.append('    </body>')
    return out


def white_prints(name, cover, spine):
    return [f'<geom name="{name}_{part}" class="arena_visual" type="mesh" mesh="white_{part}" '
            f'mass="0" material="white_{tex}"/>'
            for part, tex in (('spine', spine), ('cover_p', cover), ('cover_n', cover))]


def build_world(share, seed=None, rnd=1, book_mu=BOOK_MU, layout=None):
    # An empty ERC_SEED means 'unseeded'; int('') would otherwise abort the
    # whole launch with a traceback.
    if seed:
        random.seed(int(seed))
    venue = os.environ.get('ERC_VENUE', '')
    if venue not in ('', 'mbr'):
        sys.exit(f'ERC_VENUE is mbr or unset, not {venue!r}')
    v_bays = [int(b) for b in os.environ.get('ERC_VENUE_BAYS', '2').split(',') if b.strip()]
    v_labels = dict((int(a), int(b)) for a, b in (x.split(':') for x in
                    os.environ.get('ERC_VENUE_LABELS', '2:1').split(',') if x.strip()))
    v_style = os.environ.get('ERC_VENUE_LABEL_STYLE', 'card')
    v_wall = os.environ.get('ERC_VENUE_WALL', 'left')
    # ERC_VENUE_WHITE=bay:cover (2:A is "FUTURE"): one white book with its word,
    # for a phase 2 run at the venue
    vw = os.environ.get('ERC_VENUE_WHITE', '')
    v_white = None
    if venue and vw:
        parts = vw.split(':')       # bay:cover[:gap m from the row's coloured book]
        v_white = (int(parts[0]), parts[1]) + ((float(parts[2]),) if len(parts) > 2 else ())
    n_clutter = int(os.environ.get('ERC_CLUTTER') or 0)
    # ERC_OPEN: all (or 1) takes every wall away, sides the two along the
    # robot's sides as it faces the shelf (north and south): an open hall
    open_mode = os.environ.get('ERC_OPEN', '')
    is_open = open_mode not in ('', '0', 'sides')
    sides_open = open_mode == 'sides'
    # the venue's own draws on generators of their own: the books' draw from
    # ERC_SEED is unchanged by them
    v_rng = random.Random(int(seed or 0) * 7919 + 17)
    c_rng = random.Random(int(seed or 0) * 104729 + n_clutter)

    models = os.path.join(share, 'models')
    table_mesh = os.path.join(models, 'table', 'meshes', 'erc_base_table.STL')
    shelf_mesh = os.path.join(models, 'shelf', 'meshes', 'erc_base_shelf.STL')
    bin_mesh = os.path.join(models, 'collection_bin', 'meshes',
                            'erc_base_collection_bin.STL')
    floor_tex = os.path.join(share, 'materials', 'textures', 'arena_floor.png')

    # The table and the bin standing on it, where the layout draw puts them:
    # the bin keeps its place on the table, turned with it.
    tx, ty, tyaw = draw_layout(layout)[:3]
    table_pos = (tx, ty, TABLE_POSE[2])
    table_quat = _yawed(PROP_QUAT, tyaw)
    bx, by = BIN_POSE[0] - TABLE_POSE[0], BIN_POSE[1] - TABLE_POSE[1]
    c, s_ = math.cos(tyaw), math.sin(tyaw)
    bin_pos = (tx + c * bx - s_ * by, ty + s_ * bx + c * by, BIN_POSE[2])
    marker_tex_dir = os.path.join(models, 'number_marker', 'textures')

    for path in (table_mesh, shelf_mesh, bin_mesh, floor_tex):
        if not os.path.exists(path):
            sys.exit(f'missing arena asset: {path}')

    # ── Assets ──
    assets = [
        '    <texture type="skybox" builtin="flat" rgb1="0.8 0.8 0.8" '
        'rgb2="0.8 0.8 0.8" width="512" height="512"/>',
        f'    <texture type="2d" name="arena_floor" file="{floor_tex}"/>',
        '    <material name="arena_floor" texture="arena_floor" '
        'texrepeat="1 1" texuniform="false"/>',
        f'    <mesh name="erc_table" file="{table_mesh}"/>',
        f'    <mesh name="erc_shelf" file="{shelf_mesh}"/>',
        f'    <mesh name="erc_collection_bin" file="{bin_mesh}"/>',
    ]
    for colour, rgba in BOOK_COLOURS.items():
        assets.append(f'    <material name="book_{colour}" rgba="{fmt(rgba)}"/>')
    if rnd == 2 or v_white:
        assets.append('    <material name="book_white" rgba="1 1 1 1"/>')
        white_tex_dir = os.path.join(models, 'white_book', 'textures')
        for tex_name in sorted({t for pair in WHITE_BOOKS for t in pair}):
            tex = os.path.join(white_tex_dir, f'{tex_name}.png')
            if not os.path.exists(tex):
                sys.exit(f'missing white book texture: {tex}')
            assets.append(f'    <texture type="2d" name="white_{tex_name}" file="{tex}"/>')
            assets.append(f'    <material name="white_{tex_name}" texture="white_{tex_name}" '
                          f'texrepeat="1 1" texuniform="false"/>')
        assets += print_meshes()
    for number in range(1, NUM_COLUMNS + 1):
        tex = os.path.join(marker_tex_dir, f'{number}.png')
        if not os.path.exists(tex):
            sys.exit(f'missing number marker texture: {tex}')
        assets.append(f'    <texture type="2d" name="marker_{number}" file="{tex}"/>')
        assets.append(f'    <material name="marker_{number}" texture="marker_{number}" '
                      f'texrepeat="1 1" texuniform="false"/>')

    venue_frame, venue_lib = venue_shelf(v_wall, v_rng) if venue else ([], [])
    # the start zone is painted where the robot starts (at the origin but for
    # an 'anywhere' draw)
    start_zone = (draw_layout(layout)[3:5] if os.environ.get('ERC_LAYOUT_MODE') == 'anywhere'
                  else (0.0, 0.0))
    if venue:
        # glossy white tiles: the floor mirrors the wall, as it photographed
        assets.append('    <texture type="2d" name="venue_floor" builtin="checker" rgb1="0.93 0.93 0.92" '
                      'rgb2="0.9 0.9 0.89" width="512" height="512" mark="edge" markrgb="0.8 0.8 0.8"/>')
        assets.append('    <material name="venue_floor" texture="venue_floor" texrepeat="10 10" '
                      'reflectance="0.3" specular="0.6"/>')
        m_assets, m_geoms = venue_markers(v_labels, v_style, marker_tex_dir)
        assets += m_assets

    # ── Static arena shell ──
    body = [
        '    <!-- The arena\'s four directional lights. All four are needed, not just',
        '         an overhead one: a light pointing straight down leaves every',
        '         vertical face unlit, which renders the number markers black. -->',
        '    <light name="light_front" pos="0 -10 5" dir="0 1 -0.5" directional="true" '
        'diffuse="0.3 0.3 0.3" specular="0.05 0.05 0.05" castshadow="false"/>',
        '    <light name="light_back" pos="0 10 5" dir="0 -1 -0.5" directional="true" '
        'diffuse="0.3 0.3 0.3" specular="0.05 0.05 0.05" castshadow="false"/>',
        '    <light name="light_left" pos="-10 0 5" dir="1 0 -0.5" directional="true" '
        'diffuse="0.3 0.3 0.3" specular="0.05 0.05 0.05" castshadow="false"/>',
        '    <light name="light_right" pos="10 0 5" dir="-1 0 -0.5" directional="true" '
        'diffuse="0.3 0.3 0.3" specular="0.05 0.05 0.05" castshadow="false"/>',
        '',
        '    <!-- Ground plane, centred on the arena. The size',
        '         bounds what is drawn; a MuJoCo plane is infinite for collision. -->',
        '    <geom name="arena_floor" type="plane" pos="1 0 0" size="5 5 0.05" '
        + ('material="venue_floor"/>' if venue else 'material="arena_floor"/>'),
        '    <!-- Painted arena boundary and start zone: visual only. -->',
        '' if venue else '    <geom name="arena_boundary" class="arena_visual" type="box" pos="1 0 0.001" '
        'size="3.5 3.5 0.001" rgba="0.75 0.75 0.72 1"/>',
        f'    <geom name="start_zone" class="arena_visual" type="box" pos="{fmt(start_zone)} 0.002" '
        'size="0.4 0.4 0.001" rgba="0.1 0.4 0.1 1"/>',
        '',
        '    <!-- Outer walls. -->',
        *([] if is_open or sides_open else [
            '    <geom name="wall_north" type="box" pos="1 5 1.5" size="5 0.025 1.5" '
            'rgba="0.8 0.8 0.8 1"/>',
            '    <geom name="wall_south" type="box" pos="1 -5 1.5" size="5 0.025 1.5" '
            'rgba="0.8 0.8 0.8 1"/>']),
        *([] if is_open else [
            '    <geom name="wall_east" type="box" pos="6 0 1.5" size="0.025 5 1.5" '
            'rgba="0.8 0.8 0.8 1"/>',
            '    <geom name="wall_west" type="box" pos="-4 0 1.5" size="0.025 5 1.5" '
            'rgba="0.8 0.8 0.8 1"/>']),
        '',
        '    <!-- Table: visual mesh, collision decomposed into exact boxes. -->',
        f'    <body name="erc_table" pos="{fmt(table_pos)}" quat="{table_quat}">',
        visual_mesh('erc_table', 'erc_table', (1, 1, 1, 1)),
        collision_geoms(table_mesh, 'erc_table'),
        '    </body>',
        '',
        *(['    <!-- Shelf: the venue\'s library wall (ERC_VENUE=mbr). -->',
            '    <body name="erc_shelf" pos="0 0 0">', *venue_frame, '    </body>', *venue_lib]
          if venue else [
            '    <!-- Shelf: five 1.0 m columns over six 0.33 m rows. -->',
            f'    <body name="erc_shelf" pos="{SHELF_X} {SHELF_Y} {SHELF_Z}" quat="{PROP_QUAT}">',
            visual_mesh('erc_shelf', 'erc_shelf', (1, 1, 1, 1)),
            collision_geoms(shelf_mesh, 'erc_shelf'),
            '    </body>']),
        *clutter(n_clutter, c_rng, start=draw_layout(layout)[3:5], table=draw_layout(layout)[:2]),
        '',
        '    <!-- Collection bin: a free body, resting on the table. It never sleeps',
        '         so that a book placed in it never does either (a sleeping body has',
        '         no contacts with anything static or asleep, and /bin_contacts is',
        '         how a placement is judged). The books on the shelf do sleep. -->',
        f'    <body name="erc_collection_bin" pos="{fmt(bin_pos)}" quat="{table_quat}" sleep="never">',
        '      <freejoint name="erc_collection_bin_joint"/>',
        f'      <inertial pos="{BIN_COM}" mass="{BIN_MASS:g}" '
        f'fullinertia="{BIN_INERTIA}"/>',
        visual_mesh('erc_collection_bin', 'erc_collection_bin', (1, 0, 0, 1)),
    ]
    for name, pos, half in BIN_COLLISION:
        body.append(f'      <geom name="{name}" class="arena_collision" type="box" '
                    f'pos="{fmt(pos)}" size="{fmt(half)}"/>')
    body.append('    </body>')
    body.append('')

    if venue:
        body.append(f'    <!-- Books: bays {v_bays} set up; labels {v_labels} ({v_style}). -->')
        body += venue_books(v_bays, book_mu, v_white)
        body += m_geoms
        body.append('')
    else:
        # ── Books and markers: the draw order is part of the arena spec, so a given
        #    ERC_SEED reproduces the organisers' layout. Round 1 (upstream
        #    simulation_round_1.launch.py): five columns each shuffle their four
        #    colours then jitter each book's y; then the markers are shuffled.
        #    Round 2 (simulation_round_2.launch.py): the white books' order is
        #    shuffled first, then the columns draw as round 1 except that column 2
        #    draws no shuffle (all white) and column 4 no jitter (flush to the
        #    walls); the markers are NOT shuffled (column n carries n). ──
        if rnd == 2:
            white_books = list(WHITE_BOOKS)
            random.shuffle(white_books)
        pitch = ROUND2_PITCH if rnd == 2 else COLUMN_WIDTH
        offsets = [((NUM_COLUMNS - 1) / 2 - c) * pitch for c in range(NUM_COLUMNS)]
        body.append(f'    <!-- Books: round {rnd}, one per active shelf row, drawn from ERC_SEED. -->')
        for col in range(NUM_COLUMNS):
            shelf_num = col + 1
            white = rnd == 2 and shelf_num == WHITE_COLUMN
            edge = rnd == 2 and shelf_num == EDGE_COLUMN
            if white:
                colours_this_column = ['white'] * len(ACTIVE_ROWS)
            else:
                colours_this_column = list(BOOK_COLOURS.keys())
                random.shuffle(colours_this_column)
            for i, row in enumerate(ACTIVE_ROWS):
                colour = colours_this_column[i]
                if edge:
                    # left, right, left, right from the top row, as upstream; flush
                    # against the mesh's own divider face (see SHELF_DIVIDER_PITCH)
                    side = 1 if i % 2 == 0 else -1
                    centre = SHELF_Y + ((NUM_COLUMNS - 1) / 2 - col) * SHELF_DIVIDER_PITCH
                    y = centre + side * (COLUMN_WIDTH / 2 - BOOK_HALF[1] - EDGE_CLEARANCE)
                else:
                    y = (SHELF_Y + offsets[col]
                         + random.uniform(-COLUMN_JITTER_RANGE, COLUMN_JITTER_RANGE))
                z = SHELF_Z + ROW_FLOOR_Z_OFFSETS[row]
                x = SHELF_X - 0.1
                if white:
                    cover, spine = white_books[i]
                    name = f'book_col_{shelf_num}_row_{row + 1}_white_{cover}'
                    body += book_body(name, (x, y, z), 'book_white', book_mu,
                                      white_prints(name, cover, spine))
                else:
                    name = f'book_col_{shelf_num}_row_{row + 1}_{colour}'
                    body += book_body(name, (x, y, z), f'book_{colour}', book_mu)
        body.append('')

        # ── Number markers: static, visual only. ──
        shelf_number_markers = list(range(1, NUM_COLUMNS + 1))
        if rnd == 2:
            body.append('    <!-- Number markers on the top row: column n carries n (round 2). -->')
        else:
            body.append('    <!-- Number markers on the top row, order drawn from ERC_SEED. -->')
            random.shuffle(shelf_number_markers)
        for col in range(NUM_COLUMNS):
            number = shelf_number_markers[col]
            y = SHELF_Y + offsets[col]
            body.append(
                f'    <geom name="number_marker_col_{col + 1}" class="arena_visual" type="plane" '
                f'pos="{NUMBER_MARKER_PLATE_X:g} {y:g} {NUMBER_MARKER_PLATE_Z:g}" '
                f'xyaxes="{MARKER_XYAXES}" size="{fmt(MARKER_HALF)}" material="marker_{number}"/>')

    # A fixed vantage on the whole working area, for recordings and for a
    # human watching a headless run. Placed south of the arena looking north
    # so the table, the start zone and the shelf are all in frame.
    body.append('')
    body.append('    <camera name="spectator" mode="fixed" pos="1 -4.5 2.6" '
                'xyaxes="1 0 0 0 0.3714 0.9284" resolution="960 540"/>')

    return f'''<mujoco model="erc_arena">
  <!-- GENERATED by erc_bringup/scripts/generate_mujoco_world.py - do not edit. -->

  <!-- Sized for the working volume around the robot, not the whole 10 m arena:
       the camera near plane is a fraction of this extent, so a large extent
       would push it out past the table and clip everything the robot reaches. -->
  <statistic center="1 0 0.8" extent="2"/>

  <!-- The robot's own <option> repeats these and wins on merge (raw_inputs are
       appended after the scene). They are here so the arena is also correct
       when loaded on its own: under the Euler/pyramidal defaults a resting
       book sinks 6.4 mm into a shelf, against 0.1 mm with these. noslip 100:
       see the robot's (a held book crept 14.5 deg in the pads at 5). -->
  <option integrator="implicitfast" cone="elliptic" impratio="10" noslip_iterations="100">
    <flag multiccd="enable" sleep="enable"/>
  </option>

  <!-- The offscreen framebuffer defaults to 640x480, which is smaller than the
       spectator camera; anything rendering it offscreen fails without this.
       Merges with the robot's own <visual>, which sets the near/far planes. -->
  <visual>
    <global offwidth="960" offheight="540"/>
    <!-- The arena's nominal ambient of 0.7 cannot be copied across literally:
         MuJoCo sums ambient and diffuse differently, and four directional
         lights at the nominal 0.5 plus that ambient saturate every upward face
         to pure white. Scaled so a lit floor lands near 0.8, not past 1.0. -->
    <headlight ambient="0.25 0.25 0.25" diffuse="0.05 0.05 0.05" specular="0 0 0"/>
  </visual>

  <default>
    <default class="arena_visual">
      <geom group="2" contype="0" conaffinity="0"/>
    </default>
    <default class="arena_collision">
      <geom group="3" rgba="0 0 0 0"/>
    </default>
  </default>

  <asset>
{chr(10).join(assets)}
  </asset>

  <worldbody>
{chr(10).join(body)}
  </worldbody>
</mujoco>
'''


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-o', '--output')
    ap.add_argument('--seed', default=os.environ.get('ERC_SEED'),
                    help='book/marker layout seed (default: $ERC_SEED)')
    ap.add_argument('--round', type=int, choices=(1, 2),
                    default=int(os.environ.get('ERC_ROUND') or 1),
                    help='competition round layout (default: $ERC_ROUND or 1)')
    ap.add_argument('--book-mu', type=float,
                    default=float(os.environ.get('ERC_BOOK_MU') or BOOK_MU),
                    help=f'book sliding friction (default: $ERC_BOOK_MU or {BOOK_MU})')
    ap.add_argument('--layout', default=os.environ.get('ERC_LAYOUT'),
                    help='table and start draw (default: $ERC_LAYOUT; unset: the arena spec)')
    ap.add_argument('--print-start', action='store_true',
                    help='print the drawn start pose "x y z" and yaw, and exit')
    args = ap.parse_args()
    if args.print_start:
        sx, sy, syaw = draw_layout(args.layout)[3:]
        print(f'{sx:.4f} {sy:.4f} 0 {syaw:.4f}')
        return
    if not args.output:
        ap.error('-o/--output is required')

    share = get_package_share_directory('erc_description')
    xml = build_world(share, args.seed, args.round, args.book_mu, args.layout)
    with open(args.output, 'w') as f:
        f.write(xml)
    print(f'[generate_mujoco_world] wrote {args.output} '
          f'({len(xml):,} bytes, seed={args.seed}, round={args.round}, book_mu={args.book_mu:g}, '
          f'layout={args.layout})')


if __name__ == '__main__':
    main()
