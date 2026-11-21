"""Raw-colour and false-bin regressions from the 17 November camera frames.

The PNGs include existing annotations; tests never treat the outline or timestamp
as a target. Synthetic depth checks do not establish physical bin delivery.
"""
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from test_regressions import ROOT, PerceptionNode
from ros_stubs import Pub

# Reuse the already imported node module without installing ROS on the test host.
colour_boxes = PerceptionNode.process_bin.__globals__['colour_boxes']
bin_colour_candidates = PerceptionNode.process_bin.__globals__['bin_colour_candidates']
bin_surface_point = PerceptionNode.process_bin.__globals__['bin_surface_point']


def camera_info(width=320, height=240, focal=300.):
    return NS(width=width, height=height,
              k=[focal, 0., width/2, 0., focal, height/2, 0., 0., 1.])


def test_recorded_person_is_not_a_bin():
    image = cv2.imread(str(ROOT/'test/data/bin_false_person.png'))
    assert image is not None
    boxes, _ = bin_colour_candidates(image, image.shape[0]*image.shape[1]*.006)
    assert boxes == []


def test_existing_bin_frame_retains_the_actual_bin_candidate():
    image = cv2.imread(str(ROOT/'erc_images/collection_bin_20261109_191751_525773.png'))
    assert image is not None
    boxes, _ = bin_colour_candidates(image, image.shape[0]*image.shape[1]*.006)
    assert any(x > 350 and y > 200 for x, y, _, _ in boxes)


@pytest.mark.parametrize('filename,colour,centre', [
    ('shelf_colours_live.png', 'red', (322, 310)),
    ('shelf_colours_live.png', 'green', (260, 212)),
    ('shelf_colours_live.png', 'yellow', (347, 381)),
    ('shelf_colours_live.png', 'blue', (650, 380)),
    ('book_colours_live.png', 'red', (47, 550)),
    ('book_colours_live.png', 'green', (534, 530)),
    ('book_colours_live.png', 'yellow', (108, 634)),
    ('book_colours_live.png', 'blue', (1130, 520)),
])
def test_supplied_book_frames_keep_visible_spine_colours(filename, colour, centre):
    image = cv2.imread(str(ROOT/'test/data'/filename))
    assert image is not None
    original = image.copy()
    boxes = colour_boxes(image, colour)
    u, v = centre
    assert any(x <= u < x+w and y <= v < y+h for x, y, w, h in boxes)
    assert np.array_equal(image, original)


def test_annotated_selected_red_spine_passes_raw_colour_thresholds():
    image = cv2.imread(str(ROOT/'test/data/book_colours_live.png'))
    source = colour_boxes.__globals__
    boxes, _ = source['_raw_boxes'](image, source['STRICT']['red'], 60)
    # The saved green rectangle surrounds this spine and contaminates its
    # shelf-context ring. That overlay is absent from the live detector input,
    # so only the raw colour stage can be replayed faithfully for this spine.
    assert any(x <= 581 < x+w and y <= 620 < y+h for x, y, w, h in boxes)


@pytest.mark.parametrize('colour,bgr', [
    ('red', (30, 30, 200)), ('green', (70, 150, 30)),
    ('yellow', (20, 170, 210)), ('blue', (200, 120, 20)),
])
@pytest.mark.parametrize('background', [85, 245])
def test_flat_frame_keeps_original_colours(colour, bgr, background, monkeypatch):
    image = np.full((160, 240, 3), background, np.uint8)
    image[40:130, 90:110] = bgr
    original = image.copy()
    original_raw_boxes = colour_boxes.__globals__['_raw_boxes']
    seen = []

    def check_pixels(pixels, bands, minimum_area):
        assert np.array_equal(pixels, original)
        seen.append(True)
        return original_raw_boxes(pixels, bands, minimum_area)

    monkeypatch.setitem(colour_boxes.__globals__, '_raw_boxes', check_pixels)
    boxes = colour_boxes(image, colour)
    assert seen
    assert (90, 40, 20, 90) in boxes
    assert np.array_equal(image, original)


def bin_scene():
    image = np.full((240, 320, 3), 230, np.uint8)
    image[70:170, 60:260] = (15, 15, 220)
    depth = np.full(image.shape[:2], 8.4, np.float32)
    depth[70:170, 60:260] = .9
    # Optical forward becomes odom horizontal; bin surface height is 0.85 m.
    transform = np.array([[0., 0., 1., 0.], [-1., 0., 0., 0.],
                          [0., -1., 0., .85], [0., 0., 0., 1.]])
    return image, depth, transform


def test_bin_depth_comes_from_red_surface_even_with_background_in_centre():
    image, depth, transform = bin_scene()
    image[100:140, 140:180] = 230
    depth[100:140, 140:180] = 8.4
    boxes, mask = bin_colour_candidates(image, 200.)
    assert len(boxes) == 1
    point = bin_surface_point(boxes[0], mask, depth, transform, camera_info())
    assert point == pytest.approx([.9, .0015, .8515], abs=.01)


@pytest.mark.parametrize('bad_depth', [0., np.nan, np.inf, 8.431])
def test_bin_rejects_missing_or_distant_depth(bad_depth):
    image, depth, transform = bin_scene()
    depth[:] = bad_depth
    boxes, mask = bin_colour_candidates(image, 200.)
    assert bin_surface_point(boxes[0], mask, depth, transform, camera_info()) is None


def test_bin_rejects_incoherent_depth_and_implausible_size():
    image, depth, transform = bin_scene()
    boxes, mask = bin_colour_candidates(image, 200.)
    depth[:, :160] = .5
    depth[:, 160:] = 1.4
    assert bin_surface_point(boxes[0], mask, depth, transform, camera_info()) is None
    depth[:] = .9
    assert bin_surface_point(boxes[0], mask, depth, transform, camera_info(focal=50.)) is None


def bin_node():
    node = PerceptionNode.__new__(PerceptionNode)
    node.image, node.depth_image, node.camera_to_odom = bin_scene()
    node.info = camera_info()
    node.bin_candidate = node.bin_candidate_stamp = node.bin_stamp = None
    node.warn = lambda *args: None
    node.get_logger = lambda: NS(info=lambda *args: None)
    node.bin_hits = 0
    node.mode = 'bin'
    node.get_parameter = lambda name: NS(value=.006)
    node.make_pose = lambda point, heading: point.copy()
    node.bin_pub = Pub()
    node.reported = set()
    node.saved = []
    node.save = lambda *args: node.saved.append(args)
    node.observation_stamp = NS(sec=10, nanosec=0)
    return node


def test_bin_requires_three_distinct_consistent_frames():
    node = bin_node()
    node.process_bin()
    node.process_bin()  # Duplicate camera timestamp does not add a vote.
    assert node.bin_hits == 1
    node.observation_stamp.nanosec = 125000000
    node.process_bin()
    assert not node.bin_pub.messages and not node.saved
    node.observation_stamp.nanosec = 250000000
    node.process_bin()
    assert len(node.bin_pub.messages) == len(node.saved) == 1


def test_bin_confirmation_resets_on_loss_and_mode_change():
    node = bin_node()
    node.process_bin()
    node.depth_image[:] = 8.4
    node.observation_stamp.nanosec = 125000000
    node.process_bin()
    assert node.bin_hits == 0 and node.bin_candidate is None
    node.depth_image[:] = .9
    node.observation_stamp.nanosec = 250000000
    node.process_bin()
    assert node.bin_hits == 1
    node.on_mode(NS(data=True))  # Navigation republishes its mode every tick.
    assert node.bin_hits == 1
    node.on_mode(NS(data=False))
    assert node.bin_hits == 0 and node.bin_candidate is None


def test_bin_rejects_ambiguous_red_objects():
    node = bin_node()
    node.image[:] = 230
    node.image[50:105, 20:140] = (15, 15, 220)
    node.image[130:185, 180:300] = (15, 15, 220)
    node.depth_image[:] = .9
    for stamp in (0, 125000000, 250000000):
        node.observation_stamp.nanosec = stamp
        node.process_bin()
    assert not node.bin_pub.messages and node.bin_hits == 0
