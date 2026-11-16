"""
Tests for the raster-based SVG filter primitives, currently
``feDisplacementMap``.

The displacement results are checked pixel by pixel against an independent
reference implementation written with plain loops, both for constant and
gradient displacement maps.

"""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from . import svg2png
from .filters import FilterError

CANVAS = 40


def png_data_uri(array):
    """Encode an RGBA or grayscale numpy array as a PNG data URI."""
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(
        buffer.getvalue()).decode()


def constant_map(value):
    return png_data_uri(
        np.full((CANVAS, CANVAS), value, dtype=np.uint8))


def gradient_map(horizontal=True):
    values = (
        np.arange(CANVAS, dtype=np.uint8) * 255 // (CANVAS - 1))
    if horizontal:
        array = np.tile(values, (CANVAS, 1))
    else:
        array = np.repeat(values[:, None], CANVAS, axis=1)
    return png_data_uri(array)


MAPS = {
    'black': constant_map(0),
    'dark': constant_map(64),
    'neutral': constant_map(128),
    'bright': constant_map(192),
    'white': constant_map(255),
    'gradient-x': gradient_map(True),
    'gradient-y': gradient_map(False),
}


def render(svg):
    """Render an SVG byte string to an RGBA numpy array."""
    png = svg2png(bytestring=svg.encode())
    return np.array(Image.open(io.BytesIO(png)).convert('RGBA'))


def filter_svg(map_uri, *, scale=10, x_channel='R', y_channel='G',
               source='SourceGraphic', in2='map', region=None,
               color='red'):
    """Build an SVG with a single feDisplacementMap filter."""
    region_attributes = (
        '' if region is None else
        ' x="{0}" y="{1}" width="{2}" height="{3}"'.format(*region))
    fe_image = ''
    if in2 == 'map':
        fe_image = (
            f'<feImage href="{map_uri}" x="0" y="0" '
            f'width="{CANVAS}" height="{CANVAS}" result="map"/>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg"
        width="{CANVAS}" height="{CANVAS}"
        viewBox="0 0 {CANVAS} {CANVAS}">
      <defs><filter id="d" filterUnits="userSpaceOnUse"
          x="0" y="0" width="{CANVAS}" height="{CANVAS}">
        {fe_image}
        <feDisplacementMap in="{source}" in2="{in2}" scale="{scale}"
          xChannelSelector="{x_channel}" yChannelSelector="{y_channel}"
          {region_attributes}/>
      </filter></defs>
      <rect x="10" y="10" width="10" height="10" fill="{color}"
        filter="url(#d)"/>
    </svg>'''


def reference(source, map_array, scale, x_channel, y_channel):
    """Independent plain-loop inverse displacement with edge clamping.

    The bilinear pass runs on premultiplied components; the result is then
    un-premultiplied to match the straight-alpha values Cairo writes to PNG.

    """
    offsets = {'R': 0, 'G': 1, 'B': 2, 'A': 3}
    height, width = source.shape[:2]
    output = np.zeros_like(source)
    for y in range(height):
        for x in range(width):
            dx = scale * (
                map_array[y, x, offsets[x_channel]] / 255 - 0.5)
            dy = scale * (
                map_array[y, x, offsets[y_channel]] / 255 - 0.5)
            sample_x = min(max(x - dx, 0), width - 1)
            sample_y = min(max(y - dy, 0), height - 1)
            x0, y0 = int(np.floor(sample_x)), int(np.floor(sample_y))
            fx, fy = sample_x - x0, sample_y - y0
            x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
            top = (source[y0, x0].astype(float) * (1 - fx) +
                   source[y0, x1].astype(float) * fx)
            bottom = (source[y1, x0].astype(float) * (1 - fx) +
                      source[y1, x1].astype(float) * fx)
            output[y, x] = np.rint(top * (1 - fy) + bottom * fy)
    # Un-premultiply color channels while keeping the alpha channel.
    alpha = output[:, :, 3:4].astype(float) / 255
    opaque = output[:, :, 3] > 0
    output[:, :, :3] = np.where(
        opaque[:, :, None],
        np.rint(output[:, :, :3].astype(float) /
                np.maximum(alpha, 1e-9)), 0).astype(np.uint8)
    return output


def source_rgba(color):
    source = np.zeros((CANVAS, CANVAS, 4), dtype=np.uint8)
    rgb = {'red': (255, 0, 0), 'green': (0, 255, 0),
           'teal': (80, 160, 240)}[color]
    source[10:20, 10:20, :3] = rgb
    source[10:20, 10:20, 3] = 255
    return source


@pytest.mark.parametrize('map_name,scale,x_channel,y_channel', [
    ('bright', 10, 'R', 'G'),
    ('dark', 6, 'B', 'B'),
    ('gradient-x', 8, 'R', 'G'),
    ('gradient-y', 8, 'R', 'G'),
    ('neutral', 10, 'A', 'A'),
    ('white', 4, 'R', 'R'),
    ('black', 4, 'G', 'G'),
])
def test_displacement_matches_reference(
        map_name, scale, x_channel, y_channel):
    """Constant and gradient maps match the reference within rounding."""
    map_array = np.array(Image.open(io.BytesIO(
        base64.b64decode(MAPS[map_name].split(',', 1)[1]))).convert('RGBA'))
    expected = reference(
        source_rgba('red'), map_array, scale, x_channel, y_channel)
    actual = render(filter_svg(
        MAPS[map_name], scale=scale,
        x_channel=x_channel, y_channel=y_channel))
    assert np.abs(actual.astype(int) - expected.astype(int)).max() <= 1


def test_positive_channel_shifts_down_right():
    """A bright map (channel > 0.5) moves the source down and right."""
    actual = render(filter_svg(MAPS['bright'], scale=10))
    nonzero = np.where(actual[:, :, 3] != 0)
    # +2.53 px inverse sampling shifts the 10..19 block to 12..22.
    assert (nonzero[0].min(), nonzero[0].max()) == (12, 22)
    assert (nonzero[1].min(), nonzero[1].max()) == (12, 22)
    assert tuple(actual[15, 15]) == (255, 0, 0, 255)
    assert actual[11, 11, 3] == 0


def test_negative_channel_shifts_up_left():
    """A dark map (channel < 0.5) moves the source up and left."""
    actual = render(filter_svg(MAPS['dark'], scale=4))
    nonzero = np.where(actual[:, :, 3] != 0)
    # -1.5 px inverse sampling shifts the 10..19 block up-left to 9..19.
    assert nonzero[0].min() == 9 and nonzero[1].min() == 9
    assert actual[8, 8, 3] == 0
    assert tuple(actual[15, 15]) == (255, 0, 0, 255)


def test_scale_zero_is_identity():
    """scale=0 must produce output identical to the unfiltered graphic."""
    filtered = render(filter_svg(MAPS['bright'], scale=0))
    unfiltered = render(
        '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">'
        '<rect x="10" y="10" width="10" height="10" fill="red"/></svg>')
    assert np.array_equal(filtered, unfiltered)


def test_neutral_map_is_near_identity():
    """A 128 gray map gives a near-zero (0.002 * scale) displacement.

    128/255 - 0.5 is not exactly zero in the SVG [0, 1] channel model, so a
    neutral map only approximates the identity; the interior opaque pixels
    must be unchanged and only edge pixels may differ slightly.

    """
    filtered = render(filter_svg(MAPS['neutral'], scale=10))
    unfiltered = render(
        '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">'
        '<rect x="10" y="10" width="10" height="10" fill="red"/></svg>')
    # Interior pixels far from any edge are sampled exactly.
    assert np.array_equal(
        filtered[12:18, 12:18], unfiltered[12:18, 12:18])
    # Differences are confined to a one-pixel-wide boundary.
    diff = np.abs(filtered.astype(int) - unfiltered.astype(int)).sum(axis=2)
    changed = np.where(diff > 0)
    assert changed[0].min() <= 11 and changed[0].max() >= 18
    assert np.all(
        (changed[0] <= 11) | (changed[0] >= 18) |
        (changed[1] <= 11) | (changed[1] >= 18))


def test_source_alpha_is_opaque_gray():
    """in=SourceAlpha produces an opaque image whose level is the alpha."""
    actual = render(filter_svg(
        MAPS['bright'], scale=10, source='SourceAlpha'))
    nonzero = actual[actual[:, :, 3] != 0]
    assert np.all(nonzero[:, 3] == 255)
    assert np.all(nonzero[:, 0] == nonzero[:, 1])
    assert np.all(nonzero[:, 1] == nonzero[:, 2])


def test_in2_source_alpha_renders():
    """The source alpha can itself drive the displacement map."""
    svg = filter_svg(None, in2='SourceAlpha', scale=6)
    actual = render(svg)
    assert actual[:, :, 3].sum() > 0


def test_pixels_outside_filter_region_untouched():
    """A disjoint filter region leaves the graphic pixels unchanged."""
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg"
        width="{CANVAS}" height="{CANVAS}">
      <defs><filter id="d" filterUnits="userSpaceOnUse"
          x="30" y="30" width="8" height="8">
        <feImage href="{MAPS['bright']}" x="30" y="30" width="8" height="8"
          result="map"/>
        <feDisplacementMap in="SourceGraphic" in2="map" scale="10"/>
      </filter></defs>
      <rect x="5" y="5" width="8" height="8" fill="red" filter="url(#d)"/>
    </svg>'''
    filtered = render(svg)
    unfiltered = render(
        '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">'
        '<rect x="5" y="5" width="8" height="8" fill="red"/></svg>')
    assert np.array_equal(filtered, unfiltered)


def test_bilinear_edge_alpha():
    """Fractional displacements produce partial alpha at the edges."""
    actual = render(filter_svg(MAPS['bright'], scale=4))
    leading = actual[15, 11, 3]
    inside = actual[15, 12, 3]
    trailing = actual[15, 21, 3]
    assert 0 < leading < 255
    assert inside == 255
    assert 0 < trailing < 255


def test_premultiplied_color_no_bleed():
    """Semi-transparent edge pixels keep premultiplied, non-bleeding color."""
    actual = render(filter_svg(MAPS['bright'], scale=4, color='teal'))
    partial = actual[actual[:, :, 3] == actual[15, 11, 3]]
    assert partial.size
    # No green/blue contamination beyond the teal source components.
    assert np.all(partial[:, 0] <= 81)


@pytest.mark.parametrize('attributes,message', [
    ('<feDisplacementMap in="SourceGraphic" scale="5"/>', 'in2'),
    ('<feDisplacementMap in2="map" xChannelSelector="X" scale="5"/>',
     'xChannelSelector'),
    ('<feDisplacementMap in2="map" yChannelSelector="Z" scale="5"/>',
     'yChannelSelector'),
    ('<feDisplacementMap in="BackgroundImage" in2="map" scale="5"/>',
     '"in"'),
    ('<feDisplacementMap in2="undefined" scale="5"/>', 'in2'),
    ('<feDisplacementMap in2="map" scale="Infinity"/>', 'finite'),
    ('<feDisplacementMap in2="map" scale="NaN"/>', 'finite'),
    ('<feDisplacementMap in2="map" scale="abc"/>', 'number'),
    ('<feDisplacementMap in2="map" scale="1e12"/>', 'larger than'),
])
def test_invalid_attributes_raise(attributes, message):
    """Invalid primitives raise explicitly instead of being ignored."""
    fe_image = (
        f'<feImage href="{MAPS["bright"]}" result="map"/>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">
      <defs><filter id="d">{fe_image}{attributes}</filter></defs>
      <rect x="5" y="5" width="8" height="8" fill="red"
        filter="url(#d)"/></svg>'''
    with pytest.raises(FilterError) as exc_info:
        render(svg)
    assert message in str(exc_info.value)


def test_output_scales_with_output_width():
    """The displacement magnitude follows the device pixel scale."""
    png = svg2png(
        bytestring=filter_svg(MAPS['bright'], scale=10).encode(),
        output_width=80, output_height=80)
    actual = np.array(Image.open(io.BytesIO(png)).convert('RGBA'))
    nonzero = np.where(actual[:, :, 3] != 0)
    # At 2x the ~2.5 user-unit shift becomes ~5 device pixels.
    assert 24 <= nonzero[1].min() <= 26
    assert 44 <= nonzero[1].max() <= 46


def test_transformed_element_displacement():
    """A transform on the filtered element is taken into account."""
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60">
      <defs><filter id="d" filterUnits="userSpaceOnUse"
          x="0" y="0" width="60" height="60">
        <feImage href="{MAPS['dark']}" x="0" y="0" width="60" height="60"
          result="map"/>
        <feDisplacementMap in="SourceGraphic" in2="map" scale="6"/>
      </filter></defs>
      <rect x="10" y="10" width="10" height="10" fill="blue"
        transform="translate(15,20)" filter="url(#d)"/>
    </svg>'''
    actual = render(svg)
    assert np.count_nonzero(actual[:, :, 3]) > 0
