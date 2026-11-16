"""
feMorphology filter primitive tests.

"""

import io

import pytest
from PIL import Image

from . import svg2png

SVG_TEMPLATE = '''\
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <defs><filter id="m">{primitive}</filter></defs>
  {body}
</svg>
'''


def render(svg):
    """Render ``svg`` and return a RGBA image."""
    png = svg2png(bytestring=svg.encode('utf-8'))
    return Image.open(io.BytesIO(png)).convert('RGBA')


def render_with_filter(primitive, body, width=7, height=7):
    """Render ``body`` with a filter made of a single ``primitive``."""
    svg = SVG_TEMPLATE.format(
        width=width, height=height, primitive=primitive, body=body)
    image = render(svg)
    assert image.size == (width, height)
    return image


def opaque_pixels(image):
    """Return the set of coordinates of pixels with a non-zero alpha."""
    return {
        (x, y) for y in range(image.height) for x in range(image.width)
        if image.getpixel((x, y))[3]}


def test_dilate_single_pixel():
    """A single pixel dilated by 1 becomes a 3×3 square."""
    image = render_with_filter(
        '<feMorphology operator="dilate" radius="1"/>',
        '<rect x="3" y="3" width="1" height="1" fill="red" '
        'filter="url(#m)"/>')
    assert opaque_pixels(image) == {
        (x, y) for x in range(2, 5) for y in range(2, 5)}
    assert image.getpixel((4, 4)) == (255, 0, 0, 255)


def test_erode_block():
    """A 3×3 square eroded by 1 becomes a single pixel."""
    image = render_with_filter(
        '<feMorphology operator="erode" radius="1"/>',
        '<rect x="2" y="2" width="3" height="3" fill="red" '
        'filter="url(#m)"/>')
    assert opaque_pixels(image) == {(3, 3)}
    assert image.getpixel((3, 3)) == (255, 0, 0, 255)


def test_erode_removes_thin_shape():
    """A 1-pixel-wide shape fully disappears when eroded by 1."""
    image = render_with_filter(
        '<feMorphology operator="erode" radius="1"/>',
        '<rect x="3" y="3" width="1" height="1" fill="red" '
        'filter="url(#m)"/>')
    assert opaque_pixels(image) == set()


def test_dilate_semitransparent_rectangle():
    """Dilating a semi-transparent rectangle grows its alpha footprint.

    Color channels stay consistent with premultiplied compositing: once
    unpremultiplied, expanded pixels show the plain source color.

    """
    image = render_with_filter(
        '<feMorphology operator="dilate" radius="1"/>',
        '<rect x="2" y="2" width="4" height="4" fill="red" '
        'fill-opacity="0.5" filter="url(#m)"/>', width=8, height=8)
    # Original and expanded pixels share the same color and alpha
    assert image.getpixel((3, 3)) == (255, 0, 0, 128)
    assert image.getpixel((1, 3)) == (255, 0, 0, 128)
    assert image.getpixel((1, 1)) == (255, 0, 0, 128)
    # Pixels outside the dilated footprint are untouched
    assert image.getpixel((0, 0)) == (0, 0, 0, 0)
    assert opaque_pixels(image) == {
        (x, y) for x in range(1, 7) for y in range(1, 7)}


def test_erode_semitransparent_rectangle():
    """Eroding a semi-transparent rectangle shrinks its alpha footprint."""
    image = render_with_filter(
        '<feMorphology operator="erode" radius="1"/>',
        '<rect x="2" y="2" width="4" height="4" fill="red" '
        'fill-opacity="0.5" filter="url(#m)"/>', width=8, height=8)
    assert opaque_pixels(image) == {
        (x, y) for x in range(3, 5) for y in range(3, 5)}
    assert image.getpixel((3, 3)) == (255, 0, 0, 128)
    assert image.getpixel((2, 3)) == (0, 0, 0, 0)


def test_anisotropic_radius():
    """Radii apply independently on the x and y axes."""
    body = '<rect x="4" y="4" width="1" height="1" fill="red" ' \
           'filter="url(#m)"/>'
    image = render_with_filter(
        '<feMorphology operator="dilate" radius="2 0"/>', body,
        width=9, height=9)
    assert opaque_pixels(image) == {(x, 4) for x in range(2, 7)}
    image = render_with_filter(
        '<feMorphology operator="dilate" radius="0 2"/>', body,
        width=9, height=9)
    assert opaque_pixels(image) == {(4, y) for y in range(2, 7)}


def test_larger_radius():
    """A single pixel dilated by 2 becomes a 5×5 square."""
    image = render_with_filter(
        '<feMorphology operator="dilate" radius="2"/>',
        '<rect x="4" y="4" width="1" height="1" fill="red" '
        'filter="url(#m)"/>', width=9, height=9)
    assert opaque_pixels(image) == {
        (x, y) for x in range(2, 7) for y in range(2, 7)}


def test_zero_radius_unchanged():
    """A zero radius leaves the input unchanged."""
    body = '<rect x="1" y="1" width="3" height="2" fill="blue"{}/>'
    with_filter = SVG_TEMPLATE.format(
        width=6, height=6,
        primitive='<feMorphology operator="dilate" radius="0"/>',
        body=body.format(' filter="url(#m)"'))
    without_filter = SVG_TEMPLATE.format(
        width=6, height=6, primitive='', body=body.format(''))
    assert (
        svg2png(bytestring=with_filter.encode('utf-8')) ==
        svg2png(bytestring=without_filter.encode('utf-8')))


def test_default_radius_is_zero():
    """The default radius is 0: the input is left unchanged."""
    body = '<rect x="1" y="1" width="3" height="2" fill="blue"{}/>'
    with_filter = SVG_TEMPLATE.format(
        width=6, height=6,
        primitive='<feMorphology operator="dilate"/>',
        body=body.format(' filter="url(#m)"'))
    without_filter = SVG_TEMPLATE.format(
        width=6, height=6, primitive='', body=body.format(''))
    assert (
        svg2png(bytestring=with_filter.encode('utf-8')) ==
        svg2png(bytestring=without_filter.encode('utf-8')))


def test_default_operator_is_erode():
    """The default operator is erode."""
    body = '<rect x="2" y="2" width="3" height="3" fill="red" ' \
           'filter="url(#m)"/>'
    image = render_with_filter('<feMorphology radius="1"/>', body)
    assert opaque_pixels(image) == {(3, 3)}


def test_pixels_outside_filter_unchanged():
    """Pixels not covered by the filtered element are left unchanged."""
    image = render_with_filter(
        '<feMorphology operator="dilate" radius="1"/>',
        '<rect x="1" y="1" width="1" height="1" fill="red" '
        'filter="url(#m)"/>'
        '<rect x="5" y="5" width="1" height="1" fill="blue"/>')
    # The unfiltered blue pixel is still a single, fully opaque pixel
    assert image.getpixel((5, 5)) == (0, 0, 255, 255)
    assert image.getpixel((4, 5))[3] == 0
    assert image.getpixel((5, 4))[3] == 0
    assert image.getpixel((6, 6))[3] == 0


@pytest.mark.parametrize('operator', ('unknown', 'ERODE', ''))
def test_invalid_operator(operator):
    """Unknown operators raise an explicit error."""
    svg = SVG_TEMPLATE.format(
        width=4, height=4,
        primitive=f'<feMorphology operator="{operator}" radius="1"/>',
        body='<rect width="2" height="2" filter="url(#m)"/>')
    with pytest.raises(ValueError, match='Unknown feMorphology operator'):
        svg2png(bytestring=svg.encode('utf-8'))


@pytest.mark.parametrize('radius', ('-1', '1 -2', 'nan', 'inf', '-inf'))
def test_invalid_radius_value(radius):
    """Negative and non-finite radii raise an explicit error."""
    svg = SVG_TEMPLATE.format(
        width=4, height=4,
        primitive=f'<feMorphology operator="erode" radius="{radius}"/>',
        body='<rect width="2" height="2" filter="url(#m)"/>')
    with pytest.raises(ValueError, match='feMorphology radius must be'):
        svg2png(bytestring=svg.encode('utf-8'))


@pytest.mark.parametrize('radius', ('abc', '1 2 3', '1px'))
def test_invalid_radius_syntax(radius):
    """Radii that are not one or two plain numbers raise an error."""
    svg = SVG_TEMPLATE.format(
        width=4, height=4,
        primitive=f'<feMorphology operator="erode" radius="{radius}"/>',
        body='<rect width="2" height="2" filter="url(#m)"/>')
    with pytest.raises(ValueError, match='Invalid feMorphology radius'):
        svg2png(bytestring=svg.encode('utf-8'))


def test_huge_radius():
    """Absurdly large windows raise an explicit error."""
    svg = SVG_TEMPLATE.format(
        width=4, height=4,
        primitive='<feMorphology operator="dilate" radius="100000"/>',
        body='<rect width="2" height="2" filter="url(#m)"/>')
    with pytest.raises(ValueError, match='feMorphology radius is too large'):
        svg2png(bytestring=svg.encode('utf-8'))
