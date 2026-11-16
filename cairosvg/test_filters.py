"""
Tests for the ``feTile`` filter primitive.

The tests render small SVG samples to PNG and check the resulting pixels.
They cover the tile origin, two-color and semi-transparent tiles, transformed
coordinates, filter region clipping and the error handling of invalid or
unsupported definitions.

"""

import io

import cairocffi as cairo
import pytest

from . import svg2png

WIDTH = 60
HEIGHT = 30

REGION = (
    'filterUnits="userSpaceOnUse" primitiveUnits="userSpaceOnUse" '
    'x="0" y="0" width="{width}" height="{height}"')


def filter_definition(body='<feTile/>', width=WIDTH, height=HEIGHT):
    """Return a ``filter`` element covering the whole canvas."""
    return (
        '<filter id="f" '
        f'{REGION.format(width=width, height=height)}>'
        f'{body}</filter>')


def render(svg):
    """Render an SVG and return the PNG surface, its data and its stride."""
    png = svg2png(bytestring=svg.encode())
    surface = cairo.ImageSurface.create_from_png(io.BytesIO(png))
    return surface, bytes(surface.get_data()), surface.get_stride()


def pixel(data, x, y, stride):
    """Return the ``(r, g, b, a)`` byte values of a rendered pixel."""
    index = y * stride + x * 4
    blue, green, red, alpha = data[index:index + 4]
    return red, green, blue, alpha


def two_color_svg(tile, body='<feTile/>'):
    """Build a two-color tile repeated over the whole canvas.

    ``tile`` is the ``(x, y, width, height)`` rectangle of the reference tile:
    its left half is red and its right half is blue.

    """
    x, y, width, height = tile
    half = width / 2
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}"
            height="{HEIGHT}">
      <defs>{filter_definition(body)}</defs>
      <g filter="url(#f)">
        <rect x="{x}" y="{y}" width="{half}" height="{height}"
            fill="#ff0000"/>
        <rect x="{x + half}" y="{y}" width="{half}" height="{height}"
            fill="#0000ff"/>
      </g>
    </svg>'''


def expected_two_color(tile):
    """Expected pixels of a two-color tile repeated over the whole canvas.

    The tile repeats in both directions around its original position, so that
    negative indexes are needed too.

    """
    x, y, width, height = tile
    half = width / 2
    expected = {}
    for column in range(-(WIDTH // width) - 2, WIDTH // width + 2):
        for row in range(-(HEIGHT // height) - 2, HEIGHT // height + 2):
            origin_x = x + column * width
            origin_y = y + row * height
            for dx in range(int(width)):
                for dy in range(int(height)):
                    point = origin_x + dx, origin_y + dy
                    if not (0 <= point[0] < WIDTH and 0 <= point[1] < HEIGHT):
                        continue
                    expected[point] = (
                        (255, 0, 0, 255) if dx < half else (0, 0, 255, 255))
    return expected


def test_two_color_tiles():
    """Tiles are repeated horizontally and vertically from the tile origin."""
    tile = (10, 6, 8, 8)
    _, data, stride = render(two_color_svg(tile))
    for (x, y), color in expected_two_color(tile).items():
        assert pixel(data, x, y, stride) == color, f'pixel ({x}, {y})'


def test_tile_origin_is_aligned_on_svg_coordinates():
    """The tile origin does not depend on the filter primitive subregion."""
    tile = (7, 3, 6, 5)
    _, data, stride = render(two_color_svg(tile))
    for (x, y), color in expected_two_color(tile).items():
        assert pixel(data, x, y, stride) == color, f'pixel ({x}, {y})'


def test_translucent_tile():
    """Semi-transparent tiles keep their alpha over the whole region."""
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}"
            height="{HEIGHT}">
      <defs>{filter_definition()}</defs>
      <rect x="7" y="3" width="6" height="5" fill="#ff0000"
          fill-opacity="0.5" filter="url(#f)"/>
    </svg>'''
    _, data, stride = render(svg)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            # Premultiplied red at 50% opacity.
            assert pixel(data, x, y, stride) == (128, 0, 0, 128)


def test_source_alpha_input():
    """SourceAlpha tiles the alpha channel as a black image."""
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}"
            height="{HEIGHT}">
      <defs>{filter_definition('<feTile in="SourceAlpha"/>')}</defs>
      <rect x="5" y="5" width="5" height="5" fill="#00ff00"
          fill-opacity="0.5" filter="url(#f)"/>
    </svg>'''
    _, data, stride = render(svg)
    # The tile is 5x5 and repeats from (5, 5): inside the tile, the
    # semi-transparent alpha of the source is painted black.
    for y in range(HEIGHT):
        for x in range(WIDTH):
            inside = (x - 5) % 5 < 5 and (y - 5) % 5 < 5
            expected = (0, 0, 0, 128) if inside else (0, 0, 0, 0)
            assert pixel(data, x, y, stride) == expected


def test_transparent_pixels_are_kept():
    """Transparent parts of the tile stay transparent when repeated."""
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}"
            height="{HEIGHT}">
      <defs>{filter_definition()}</defs>
      <g filter="url(#f)">
        <rect x="10" y="10" width="4" height="4" fill="#ff0000"/>
      </g>
    </svg>'''
    _, data, stride = render(svg)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            # The tile is 4x4 and repeats from (10, 10), leaving the rest of
            # the canvas transparent.
            inside = (x - 10) % 4 < 4 and (y - 10) % 4 < 4
            expected = (255, 0, 0, 255) if inside else (0, 0, 0, 0)
            assert pixel(data, x, y, stride) == expected


def test_primitive_subregion_clips_tiles():
    """Tiles are limited to the filter primitive subregion."""
    body = '<feTile x="10" y="5" width="20" height="20"/>'
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}"
            height="{HEIGHT}">
      <defs>{filter_definition(body)}</defs>
      <rect x="5" y="5" width="5" height="5" fill="#00ff00"
          filter="url(#f)"/>
    </svg>'''
    _, data, stride = render(svg)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            inside = 10 <= x < 30 and 5 <= y < 25
            expected = (0, 255, 0, 255) if inside else (0, 0, 0, 0)
            assert pixel(data, x, y, stride) == expected


def test_filter_region_clips_output():
    """Pixels outside of the filter region are left unchanged."""
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}"
            height="{HEIGHT}">
      <rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>
      <defs>
        <filter id="f" filterUnits="userSpaceOnUse"
            primitiveUnits="userSpaceOnUse"
            x="10" y="5" width="20" height="20"><feTile/></filter>
      </defs>
      <rect x="10" y="10" width="5" height="5" fill="#0000ff"
          filter="url(#f)"/>
    </svg>'''
    _, data, stride = render(svg)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            inside = 10 <= x < 30 and 5 <= y < 25
            expected = (
                (0, 0, 255, 255) if inside else (255, 255, 255, 255))
            assert pixel(data, x, y, stride) == expected


def test_scale_transform():
    """The tile follows the current user coordinate system."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="96" height="48">
      <defs>
        <filter id="f" filterUnits="userSpaceOnUse"
            primitiveUnits="userSpaceOnUse"
            x="0" y="0" width="96" height="48"><feTile/></filter>
      </defs>
      <g transform="translate(8 4) scale(2 2)" filter="url(#f)">
        <rect x="4" y="4" width="2" height="4" fill="#ff0000"/>
        <rect x="6" y="4" width="2" height="4" fill="#0000ff"/>
      </g>
    </svg>'''
    _, data, stride = render(svg)
    # Device = translate(8, 4) + 2 * local coordinates, so the tile period is
    # 8 device pixels whatever the filter region is.
    for y in range(48):
        for x in range(96):
            local_x, local_y = (x - 8) / 2, (y - 4) / 2
            if local_x < 0 or local_y < 0:
                expected = (0, 0, 0, 0)
            else:
                expected = (
                    (255, 0, 0, 255) if local_x % 4 < 2
                    else (0, 0, 255, 255))
            assert pixel(data, x, y, stride) == expected


def test_rotation_transform():
    """Tiles stay aligned with the rotated user coordinate system."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80">
      <defs>
        <filter id="f" filterUnits="userSpaceOnUse"
            primitiveUnits="userSpaceOnUse"
            x="0" y="0" width="80" height="80"><feTile/></filter>
      </defs>
      <g transform="rotate(90 40 40)" filter="url(#f)">
        <rect x="20" y="20" width="4" height="8" fill="#ff0000"/>
        <rect x="24" y="20" width="4" height="8" fill="#0000ff"/>
      </g>
    </svg>'''
    _, data, stride = render(svg)
    # Inverse of rotate(90 40 40): local = (y, 80 - x). The reference tile is
    # anchored at the element bounds origin (20, 20) and is 8x8 big.
    for y in range(80):
        for x in range(80):
            local_x = y
            red = (local_x - 20) % 8 < 4
            expected = (255, 0, 0, 255) if red else (0, 0, 255, 255)
            assert pixel(data, x, y, stride) == expected


def test_empty_tile_keeps_backdrop():
    """An element without geometry produces a fully transparent result."""
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}"
            height="{HEIGHT}">
      <rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>
      <defs>{filter_definition()}</defs>
      <g filter="url(#f)"></g>
    </svg>'''
    _, data, stride = render(svg)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            assert pixel(data, x, y, stride) == (255, 255, 255, 255)


def test_zero_sized_primitive_region():
    """A zero-sized primitive subregion draws nothing."""
    body = '<feTile x="0" y="0" width="0" height="10"/>'
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}"
            height="{HEIGHT}">
      <defs>{filter_definition(body)}</defs>
      <rect x="5" y="5" width="5" height="5" fill="#ff0000"
          filter="url(#f)"/>
    </svg>'''
    _, data, stride = render(svg)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            assert pixel(data, x, y, stride) == (0, 0, 0, 0)


def test_zero_sized_filter_region():
    """A zero-sized filter region disables the element."""
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}"
            height="{HEIGHT}">
      <rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>
      <defs>
        <filter id="f" filterUnits="userSpaceOnUse"
            x="0" y="0" width="0" height="{HEIGHT}"><feTile/></filter>
      </defs>
      <rect x="5" y="5" width="5" height="5" fill="#ff0000"
          filter="url(#f)"/>
    </svg>'''
    _, data, stride = render(svg)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            assert pixel(data, x, y, stride) == (255, 255, 255, 255)


def test_negative_region_is_rejected():
    """A negative filter primitive subregion size is an error."""
    body = '<feTile x="0" y="0" width="-10" height="10"/>'
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}"
            height="{HEIGHT}">
      <defs>{filter_definition(body)}</defs>
      <rect x="5" y="5" width="5" height="5" filter="url(#f)"/>
    </svg>'''
    with pytest.raises(ValueError, match='negative size'):
        render(svg)


def test_too_many_repetitions_is_rejected():
    """An absurd number of tiles is reported instead of being allocated."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="2000"
            height="2000">
      <defs>
        <filter id="f" filterUnits="userSpaceOnUse"
            primitiveUnits="userSpaceOnUse"
            x="0" y="0" width="2000" height="2000"><feTile/></filter>
      </defs>
      <g filter="url(#f)">
        <rect x="0" y="0" width="0.000000001" height="0.000000001"
            fill="#ff0000"/>
      </g>
    </svg>'''
    with pytest.raises(ValueError, match='too many feTile repetitions'):
        render(svg)


@pytest.mark.parametrize('input_name', (
    'BackgroundImage', 'BackgroundAlpha', 'FillPaint', 'StrokePaint',
    'missing'))
def test_unsupported_input_is_rejected(input_name):
    """Unsupported or unknown ``in`` references raise a clear error."""
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">
      <defs>{filter_definition(f'<feTile in="{input_name}"/>', 40, 40)}</defs>
      <rect width="10" height="10" filter="url(#f)"/>
    </svg>'''
    with pytest.raises(ValueError, match='unsupported feTile input'):
        render(svg)


def test_intermediate_result_is_rejected():
    """``feTile`` can not consume the result of a previous primitive."""
    body = '<feFlood flood-color="#ff0000" result="flood"/><feTile/>'
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">
      <defs>{filter_definition(body, 40, 40)}</defs>
      <rect width="10" height="10" filter="url(#f)"/>
    </svg>'''
    with pytest.raises(ValueError, match='without an explicit input'):
        render(svg)


def test_fe_tile_must_be_last():
    """A filter whose output is not the ``feTile`` result is rejected."""
    body = '<feTile/><feFlood flood-color="#ff0000"/>'
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">
      <defs>{filter_definition(body, 40, 40)}</defs>
      <rect width="10" height="10" filter="url(#f)"/>
    </svg>'''
    with pytest.raises(ValueError, match='last filter primitive'):
        render(svg)


def test_unfiltered_pixels_are_unchanged():
    """Neighbouring opaque and transparent pixels are not modified."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60">
      <rect x="0" y="0" width="30" height="60" fill="#00ff00"/>
      <defs>
        <filter id="f" filterUnits="userSpaceOnUse"
            primitiveUnits="userSpaceOnUse"
            x="35" y="35" width="25" height="25"><feTile/></filter>
      </defs>
      <rect x="40" y="40" width="10" height="10" fill="#ff0000"
          filter="url(#f)"/>
    </svg>'''
    _, data, stride = render(svg)
    assert pixel(data, 10, 10, stride) == (0, 255, 0, 255)
    # (33, 33) is just outside of the filter region: the transparent backdrop
    # is kept instead of being covered by a tile.
    assert pixel(data, 33, 33, stride) == (0, 0, 0, 0)
    # Inside the region, the opaque 10x10 tile anchored at (40, 40) repeats:
    # the whole filter region becomes red.
    for x, y in ((40, 40), (45, 45), (49, 49), (50, 50), (59, 59)):
        assert pixel(data, x, y, stride) == (255, 0, 0, 255)


def test_pattern_fill_is_unchanged():
    """Regular pattern fills are not affected by ``feTile`` support."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">
      <defs>
        <pattern id="p" width="10" height="10" patternUnits="userSpaceOnUse">
          <rect width="5" height="10" fill="#ff0000"/>
          <rect x="5" width="5" height="10" fill="#0000ff"/>
        </pattern>
      </defs>
      <rect width="40" height="40" fill="url(#p)"/>
    </svg>'''
    _, data, stride = render(svg)
    for y in range(40):
        for x in range(40):
            expected = (
                (255, 0, 0, 255) if x % 10 < 5 else (0, 0, 255, 255))
            assert pixel(data, x, y, stride) == expected


def test_object_bounding_box_units():
    """``objectBoundingBox`` primitive units are fractions of the bounds."""
    body = '<feTile x="0" y="0" width="1" height="1"/>'
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}"
            height="{HEIGHT}">
      <defs>
        <filter id="f" filterUnits="objectBoundingBox"
            primitiveUnits="objectBoundingBox"
            x="0" y="0" width="1" height="1">{body}</filter>
      </defs>
      <g filter="url(#f)">
        <rect x="10" y="10" width="10" height="10" fill="#ff0000"/>
      </g>
    </svg>'''
    _, data, stride = render(svg)
    # The bounds are (10, 10, 10, 10) and the tile repeats over that region.
    for y in range(HEIGHT):
        for x in range(WIDTH):
            inside = 10 <= x < 20 and 10 <= y < 20
            expected = (255, 0, 0, 255) if inside else (0, 0, 0, 0)
            assert pixel(data, x, y, stride) == expected


def test_group_bounding_box_defines_the_tile():
    """The reference tile is the bounding box of the whole group."""
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}"
            height="{HEIGHT}">
      <defs>{filter_definition()}</defs>
      <g filter="url(#f)">
        <rect x="8" y="8" width="6" height="6" fill="#ff0000"/>
        <rect x="14" y="8" width="6" height="6" fill="#0000ff"/>
      </g>
    </svg>'''
    _, data, stride = render(svg)
    # The tile spans x=8..20, y=8..14 and repeats from the origin (8, 8).
    for y in range(HEIGHT):
        for x in range(WIDTH):
            local_x = (x - 8) % 12
            expected = (
                (255, 0, 0, 255) if local_x < 6 else (0, 0, 255, 255))
            assert pixel(data, x, y, stride) == expected


def test_default_filter_region_is_the_bounding_box_band():
    """Default filter region is -10% / 120% of the object bounding box."""
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}"
            height="{HEIGHT}">
      <defs><filter id="f"><feTile/></filter></defs>
      <rect x="10" y="10" width="10" height="10" fill="#ff0000"
          filter="url(#f)"/>
    </svg>'''
    _, data, stride = render(svg)
    # The region is (9, 9, 12, 12) for a (10, 10, 10, 10) bounding box.
    for y in range(HEIGHT):
        for x in range(WIDTH):
            inside = 9 <= x < 21 and 9 <= y < 21
            expected = (255, 0, 0, 255) if inside else (0, 0, 0, 0)
            assert pixel(data, x, y, stride) == expected


def test_huge_transform_degrades_tile_resolution():
    """A big transform keeps the tile image bounded instead of crashing."""
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}"
            height="{HEIGHT}">
      <defs>{filter_definition()}</defs>
      <g transform="scale(3000)" filter="url(#f)">
        <rect x="0.001" y="0.001" width="0.002" height="0.002"
            fill="#ff0000"/>
      </g>
    </svg>'''
    # The tile would be 6x6 user units at a 3000x scale: the intermediate
    # image is downscaled rather than raising or exhausting memory.
    surface, data, stride = render(svg)
    assert surface.get_width() == WIDTH
    assert pixel(data, WIDTH // 2, HEIGHT // 2, stride) == (255, 0, 0, 255)
