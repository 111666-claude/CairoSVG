"""
Tests for vector-effect="non-scaling-stroke" support.

"""

import io

import pytest
from PIL import Image

from . import SURFACES, svg2png

MAGIC_NUMBERS = {
    'SVG': b'<?xml',
    'PNG': b'\211PNG\r\n\032\n',
    'PDF': b'%PDF',
    'PS': b'%!',
}


def render_image(svg):
    """Render ``svg`` to a PNG image composited over a white background."""
    png = svg2png(bytestring=svg.encode())
    image = Image.open(io.BytesIO(png)).convert('RGBA')
    background = Image.new('RGBA', image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, image).convert('L')


def dark_run(image, x_range, y):
    """Count dark pixels along row ``y`` for ``x`` in ``x_range``."""
    return sum(image.getpixel((x, y)) < 128 for x in x_range)


def svg_doc(content, attributes='width="120" height="120"'):
    """Wrap ``content`` in an SVG document."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" {attributes}>'
        f'{content}</svg>')


def test_uniform_scale():
    """Stroke width is stable under a uniform scale."""
    scaled = svg_doc(
        '<g transform="scale(4)">'
        '<rect x="10" y="10" width="10" height="10" fill="none"'
        ' stroke="black" stroke-width="2"'
        ' vector-effect="non-scaling-stroke"/></g>')
    reference = svg_doc(
        '<rect x="40" y="40" width="40" height="40" fill="none"'
        ' stroke="black" stroke-width="2"/>')
    plain = svg_doc(
        '<g transform="scale(4)">'
        '<rect x="10" y="10" width="10" height="10" fill="none"'
        ' stroke="black" stroke-width="2"/></g>')
    assert render_image(scaled) == render_image(reference)
    assert render_image(plain) != render_image(reference)


def test_non_uniform_scale():
    """Stroke width is stable under a non-uniform scale."""
    scaled = svg_doc(
        '<g transform="scale(4 1)">'
        '<rect x="10" y="10" width="10" height="10" fill="none"'
        ' stroke="black" stroke-width="2"'
        ' vector-effect="non-scaling-stroke"/></g>')
    reference = svg_doc(
        '<rect x="40" y="10" width="40" height="10" fill="none"'
        ' stroke="black" stroke-width="2"/>')
    plain = svg_doc(
        '<g transform="scale(4 1)">'
        '<rect x="10" y="10" width="10" height="10" fill="none"'
        ' stroke="black" stroke-width="2"/></g>')
    assert render_image(scaled) == render_image(reference)
    assert render_image(plain) != render_image(reference)


def test_rotation():
    """Stroke width is stable under a rotation combined with a scale."""
    rotated = svg_doc(
        '<g transform="rotate(30) scale(4)">'
        '<rect x="10" y="10" width="10" height="10" fill="none"'
        ' stroke="black" stroke-width="2"'
        ' vector-effect="non-scaling-stroke"/></g>')
    reference = svg_doc(
        '<g transform="rotate(30)">'
        '<rect x="40" y="40" width="40" height="40" fill="none"'
        ' stroke="black" stroke-width="2"/></g>')
    assert render_image(rotated) == render_image(reference)


def test_nested_transforms():
    """Stroke width is stable under nested transformations."""
    nested = svg_doc(
        '<g transform="translate(5 3) scale(2)">'
        '<g transform="rotate(10) scale(3)">'
        '<rect x="10" y="10" width="10" height="10" fill="none"'
        ' stroke="black" stroke-width="2"'
        ' vector-effect="non-scaling-stroke"/></g></g>')
    flat = svg_doc(
        '<g transform="translate(5 3) scale(6) rotate(10)">'
        '<rect x="10" y="10" width="10" height="10" fill="none"'
        ' stroke="black" stroke-width="2"'
        ' vector-effect="non-scaling-stroke"/></g>')
    plain = svg_doc(
        '<g transform="translate(5 3) scale(2)">'
        '<g transform="rotate(10) scale(3)">'
        '<rect x="10" y="10" width="10" height="10" fill="none"'
        ' stroke="black" stroke-width="2"/></g></g>')
    assert render_image(nested) == render_image(flat)
    # Without vector-effect, the stroke is scaled by 6
    assert render_image(nested) != render_image(plain)


def test_stroke_width_pixels():
    """The rendered stroke keeps its specified width in device pixels."""
    scaled = svg_doc(
        '<g transform="scale(4 0.5)">'
        '<line x1="10" y1="40" x2="30" y2="40" stroke="black"'
        ' stroke-width="2" vector-effect="non-scaling-stroke"/>'
        '<line x1="20" y1="20" x2="20" y2="60" stroke="black"'
        ' stroke-width="2" vector-effect="non-scaling-stroke"/></g>')
    image = render_image(scaled)
    # Horizontal line at y = 20 (40 * 0.5): 2 device pixels high
    assert dark_run(image, range(45, 75), 20) == 30
    column = [image.getpixel((60, y)) < 128 for y in range(14, 26)]
    assert sum(column) == 2
    # Vertical line at x = 80 (20 * 4): 2 device pixels wide
    row = [image.getpixel((x, 15)) < 128 for x in range(74, 86)]
    assert sum(row) == 2


def test_curve():
    """Curves get a constant stroke width under a non-uniform scale."""
    scaled = svg_doc(
        '<g transform="scale(4 1)">'
        '<path d="M 10 20 C 10 10 30 10 30 20" fill="none" stroke="black"'
        ' stroke-width="2" vector-effect="non-scaling-stroke"/></g>')
    reference = svg_doc(
        '<path d="M 40 20 C 40 10 120 10 120 20" fill="none" stroke="black"'
        ' stroke-width="2"/>')
    assert render_image(scaled) == render_image(reference)


def test_arc():
    """Arcs get a constant stroke width under a non-uniform scale."""
    scaled = svg_doc(
        '<g transform="scale(4 1)">'
        '<circle cx="20" cy="30" r="10" fill="none" stroke="black"'
        ' stroke-width="2" vector-effect="non-scaling-stroke"/></g>')
    plain = svg_doc(
        '<g transform="scale(4 1)">'
        '<circle cx="20" cy="30" r="10" fill="none" stroke="black"'
        ' stroke-width="2"/></g>')
    # The circle becomes an ellipse centered on (80, 30) with radii
    # (40, 10): the horizontal stroke run on its left side (x = 40)
    # keeps the specified width of 2 device pixels…
    image = render_image(scaled)
    row = [image.getpixel((x, 30)) < 128 for x in range(36, 48)]
    assert sum(row) == 2
    # …while a plain stroke is scaled to 8 device pixels
    image = render_image(plain)
    row = [image.getpixel((x, 30)) < 128 for x in range(32, 48)]
    assert sum(row) == 8


def test_dash():
    """Dash patterns are computed in the host coordinate system."""
    scaled = svg_doc(
        '<g transform="scale(4)">'
        '<line x1="2" y1="10" x2="30" y2="10" stroke="black" stroke-width="2"'
        ' stroke-dasharray="6 4" vector-effect="non-scaling-stroke"/></g>')
    reference = svg_doc(
        '<line x1="8" y1="40" x2="120" y2="40" stroke="black"'
        ' stroke-width="2" stroke-dasharray="6 4"/>')
    plain = svg_doc(
        '<g transform="scale(4)">'
        '<line x1="2" y1="10" x2="30" y2="10" stroke="black" stroke-width="2"'
        ' stroke-dasharray="6 4"/></g>')
    assert render_image(scaled) == render_image(reference)
    assert render_image(plain) != render_image(reference)


def test_linecap_linejoin():
    """Line caps and joins are computed in the host coordinate system."""
    scaled = svg_doc(
        '<g transform="scale(3)">'
        '<polyline points="10,20 20,10 30,20" fill="none" stroke="black"'
        ' stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'
        ' vector-effect="non-scaling-stroke"/></g>')
    reference = svg_doc(
        '<polyline points="30,60 60,30 90,60" fill="none" stroke="black"'
        ' stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')
    assert render_image(scaled) == render_image(reference)


def test_clip_path():
    """Clipping is applied to the transformed geometry as usual."""
    scaled = svg_doc(
        '<defs><clipPath id="c">'
        '<rect x="12" y="12" width="6" height="6"/>'
        '</clipPath></defs>'
        '<g transform="scale(4)">'
        '<rect x="10" y="10" width="10" height="10" fill="none"'
        ' stroke="black" stroke-width="2" clip-path="url(#c)"'
        ' vector-effect="non-scaling-stroke"/></g>')
    reference = svg_doc(
        '<defs><clipPath id="c">'
        '<rect x="48" y="48" width="24" height="24"/>'
        '</clipPath></defs>'
        '<rect x="40" y="40" width="40" height="40" fill="none"'
        ' stroke="black" stroke-width="2" clip-path="url(#c)"/>')
    assert render_image(scaled) == render_image(reference)


def test_use():
    """vector-effect on referenced content and on use is honored."""
    referenced = svg_doc(
        '<defs><path id="p" d="M 10 10 L 30 10" fill="none" stroke="black"'
        ' stroke-width="2" vector-effect="non-scaling-stroke"/></defs>'
        '<g transform="scale(4)"><use href="#p"/></g>')
    on_use = svg_doc(
        '<defs><path id="p" d="M 10 10 L 30 10" fill="none" stroke="black"'
        ' stroke-width="2"/></defs>'
        '<g transform="scale(4)">'
        '<use href="#p" vector-effect="non-scaling-stroke"/></g>')
    reference = svg_doc(
        '<path d="M 40 40 L 120 40" fill="none" stroke="black"'
        ' stroke-width="2"/>')
    assert render_image(referenced) == render_image(reference)
    assert render_image(on_use) == render_image(reference)


def test_none_and_default():
    """vector-effect="none" and no vector-effect keep the old behavior."""
    none = svg_doc(
        '<g transform="scale(4)">'
        '<rect x="10" y="10" width="10" height="10" fill="none"'
        ' stroke="black" stroke-width="2" vector-effect="none"/></g>')
    default = svg_doc(
        '<g transform="scale(4)">'
        '<rect x="10" y="10" width="10" height="10" fill="none"'
        ' stroke="black" stroke-width="2"/></g>')
    reference = svg_doc(
        '<rect x="40" y="40" width="40" height="40" fill="none"'
        ' stroke="black" stroke-width="8"/>')
    assert render_image(none) == render_image(reference)
    assert render_image(default) == render_image(reference)


def test_unsupported_value():
    """Unsupported vector-effect values warn and are ignored."""
    unsupported = svg_doc(
        '<g transform="scale(4)">'
        '<rect x="10" y="10" width="10" height="10" fill="none"'
        ' stroke="black" stroke-width="2" vector-effect="non-rotation"'
        '/></g>')
    default = svg_doc(
        '<g transform="scale(4)">'
        '<rect x="10" y="10" width="10" height="10" fill="none"'
        ' stroke="black" stroke-width="2"/></g>')
    with pytest.warns(UserWarning, match='Unsupported vector-effect'):
        image = render_image(unsupported)
    assert image == render_image(default)


def test_viewport_boundary():
    """Shapes crossing the viewport boundary are stroked consistently."""
    scaled = svg_doc(
        '<g transform="scale(4)">'
        '<rect x="-5" y="-5" width="10" height="10" fill="none"'
        ' stroke="black" stroke-width="2"'
        ' vector-effect="non-scaling-stroke"/></g>')
    reference = svg_doc(
        '<rect x="-20" y="-20" width="40" height="40" fill="none"'
        ' stroke="black" stroke-width="2"/>')
    assert render_image(scaled) == render_image(reference)


def test_viewbox():
    """Stroke width is expressed in the viewport coordinate system."""
    scaled = svg_doc(
        '<g transform="scale(2)">'
        '<rect x="5" y="5" width="10" height="10" fill="none"'
        ' stroke="black" stroke-width="2"'
        ' vector-effect="non-scaling-stroke"/></g>',
        attributes='width="100" height="100" viewBox="0 0 50 50"')
    reference = svg_doc(
        '<rect x="20" y="20" width="40" height="40" fill="none"'
        ' stroke="black" stroke-width="4"/>',
        attributes='width="100" height="100"')
    assert render_image(scaled) == render_image(reference)


@pytest.mark.parametrize('format_name', MAGIC_NUMBERS)
def test_formats(format_name):
    """Non-scaling strokes are supported by all output formats."""
    svg = svg_doc(
        '<g transform="rotate(30) scale(4 1)">'
        '<rect x="10" y="10" width="10" height="10" fill="none"'
        ' stroke="black" stroke-width="2"'
        ' vector-effect="non-scaling-stroke"/></g>')
    content = SURFACES[format_name].convert(svg.encode())
    assert content.startswith(MAGIC_NUMBERS[format_name])
