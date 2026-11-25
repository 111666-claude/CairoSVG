"""
Test the bounding box calculation with nested transformations.

"""

import io
import math

import pytest

from . import svg2png
from .bounding_box import calculate_bounding_box
from .parser import Tree


class FakeSurface:
    """Minimal surface providing the attributes needed to compute sizes."""

    def __init__(self, width=200, height=200):
        self.width = self.context_width = width
        self.height = self.context_height = height
        self.font_size = 12


def parse_tree(svg):
    """Parse an SVG string and return its root ``Tree`` node."""
    return Tree(bytestring=svg.encode())


def find_group(tree):
    """Return the first ``g`` child of ``tree``."""
    return next(child for child in tree.children if child.tag == 'g')


def rotate_point(x, y, angle):
    """Rotate point ``(x, y)`` around origin by ``angle`` (in degrees)."""
    rad = math.radians(angle)
    return (
        x * math.cos(rad) - y * math.sin(rad),
        y * math.cos(rad) + x * math.sin(rad))


def aabb(points):
    """Get the axis-aligned bounding box of ``points``."""
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def assert_bounding_box_almost_equal(box, expected):
    """Compare two ``(x, y, width, height)`` bounding boxes."""
    assert box is not None
    assert len(box) == 4
    for value, expected_value in zip(box, expected):
        assert value == pytest.approx(expected_value, abs=1e-9)


def test_nested_groups_transform():
    """Bounding box of nested groups accumulates parent/child matrices."""
    tree = parse_tree('''
        <svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">
          <g transform="translate(50, 60)">
            <g transform="scale(2)">
              <g transform="rotate(90)">
                <rect x="10" y="-5" width="20" height="10"/>
              </g>
            </g>
          </g>
        </svg>''')
    group = find_group(tree)
    box = calculate_bounding_box(FakeSurface(), group)

    # Independent expectation: transform the rect corners by hand
    corners = [(10, -5), (30, -5), (10, 5), (30, 5)]
    corners = [rotate_point(x, y, 90) for x, y in corners]
    corners = [(2 * x, 2 * y) for x, y in corners]
    corners = [(x + 50, y + 60) for x, y in corners]
    assert_bounding_box_almost_equal(box, aabb(corners))
    assert box == pytest.approx((40, 80, 20, 40))


def test_transformed_path_in_group():
    """Bounding box of a transformed path follows the drawn coordinates."""
    tree = parse_tree('''
        <svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">
          <g transform="translate(100, 50) scale(0.5)">
            <path d="M 0 0 L 40 0 L 40 20 L 0 20 Z"/>
          </g>
        </svg>''')
    group = find_group(tree)
    box = calculate_bounding_box(FakeSurface(), group)

    corners = [(0, 0), (40, 0), (0, 20), (40, 20)]
    corners = [(0.5 * x + 100, 0.5 * y + 50) for x, y in corners]
    assert_bounding_box_almost_equal(box, aabb(corners))
    assert box == pytest.approx((100, 50, 20, 10))


def test_rotated_rect_bounding_box_is_axis_aligned():
    """Rotated shapes produce the axis-aligned box of the drawn shape."""
    tree = parse_tree('''
        <svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">
          <g transform="rotate(45, 50, 50)">
            <rect x="40" y="45" width="20" height="10"/>
          </g>
        </svg>''')
    group = find_group(tree)
    box = calculate_bounding_box(FakeSurface(), group)

    # Independent expectation: rotate corners around (50, 50) by 45 degrees
    corners = [(40, 45), (60, 45), (40, 55), (60, 55)]
    corners = [
        (50 + dx, 50 + dy)
        for dx, dy in (rotate_point(x - 50, y - 50, 45) for x, y in corners)]
    assert_bounding_box_almost_equal(box, aabb(corners))


def test_empty_group_has_no_bounding_box():
    """Empty groups don't produce fake bounding boxes."""
    tree = parse_tree('''
        <svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">
          <g transform="translate(10, 10)">
            <g transform="scale(3)"/>
          </g>
        </svg>''')
    group = find_group(tree)
    assert calculate_bounding_box(FakeSurface(), group) is None


def test_negative_coordinates():
    """Shapes at negative coordinates produce valid bounding boxes."""
    tree = parse_tree('''
        <svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">
          <g transform="translate(-40, -30)">
            <rect x="10" y="10" width="20" height="10"/>
          </g>
        </svg>''')
    group = find_group(tree)
    box = calculate_bounding_box(FakeSurface(), group)
    assert box == pytest.approx((-30, -20, 20, 10))


def test_use_bounding_box():
    """``use`` bounding boxes include x/y offsets and transformations."""
    tree = parse_tree('''
        <svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">
          <defs>
            <rect id="rect" x="0" y="0" width="10" height="10"/>
          </defs>
          <g transform="translate(100, 100)">
            <use href="#rect" x="5" y="5" transform="rotate(45)"/>
          </g>
        </svg>''')
    group = find_group(tree)
    box = calculate_bounding_box(FakeSurface(), group)

    # Referenced rect, shifted by x/y, rotated by 45 degrees around origin,
    # then translated by the group
    corners = [(5, 5), (15, 5), (5, 15), (15, 15)]
    corners = [rotate_point(x, y, 45) for x, y in corners]
    corners = [(x + 100, y + 100) for x, y in corners]
    assert_bounding_box_almost_equal(box, aabb(corners))


def test_untransformed_shapes_keep_legacy_results():
    """Bounding boxes of untransformed shapes are unchanged."""
    tree = parse_tree('''
        <svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">
          <g>
            <rect x="5" y="10" width="13" height="15"/>
            <circle cx="50" cy="60" r="7"/>
            <path d="M 0 0 L 30 0 L 30 40 Z"/>
          </g>
        </svg>''')
    group = find_group(tree)
    rect, circle, path = group.children
    surface = FakeSurface()
    assert calculate_bounding_box(surface, rect) == (5, 10, 13, 15)
    assert calculate_bounding_box(surface, circle) == (43, 53, 14, 14)
    assert calculate_bounding_box(surface, path) == (0, 0, 30, 40)
    assert calculate_bounding_box(surface, group) == (0, 0, 57, 67)


def test_repeated_calls_are_stable():
    """Repeated bounding box calculations give the same result."""
    tree = parse_tree('''
        <svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">
          <g transform="translate(50, 60) rotate(30) scale(2)">
            <rect x="10" y="10" width="20" height="10"/>
          </g>
        </svg>''')
    group = find_group(tree)
    surface = FakeSurface()
    first = calculate_bounding_box(surface, group)
    second = calculate_bounding_box(surface, group)
    assert first == second
    assert not hasattr(surface, 'context')


def test_bounding_box_matches_rendered_pixels():
    """Computed bounding box covers the rendered pixels, without inflation."""
    svg = '''
        <svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">
          <g transform="translate(60, 70)">
            <g transform="rotate(30)">
              <rect x="10" y="5" width="60" height="30" fill="#000"/>
            </g>
          </g>
        </svg>'''
    tree = parse_tree(svg)
    group = find_group(tree)
    box = calculate_bounding_box(FakeSurface(), group)

    png = svg2png(bytestring=svg.encode(), background_color='white')
    from PIL import Image
    image = Image.open(io.BytesIO(png)).convert('L')
    # Bounding box of painted (non-white) pixels, in Pillow's (l, u, r, d)
    painted = image.point(lambda pixel: 255 - pixel).getbbox()
    assert painted is not None
    minx, miny, width, height = box
    maxx, maxy = minx + width, miny + height
    left, upper, right, lower = painted

    # The computed box must not miss any painted pixel (anti-aliasing can
    # bleed one pixel outside of the geometric box)
    tolerance = 2
    assert minx <= left + tolerance
    assert miny <= upper + tolerance
    assert maxx >= right - tolerance
    assert maxy >= lower - tolerance

    # The computed box must not be inflated either
    assert left <= minx + tolerance
    assert upper <= miny + tolerance
    assert right >= maxx - tolerance
    assert lower >= maxy - tolerance
