"""
Tests for the ``feColorMatrix`` SVG filter primitive.

The filter is evaluated on a raster surface, so the tests render small SVG
samples to PNG and check the resulting pixels. Both the four defined matrix
modes and the way the primitive participates in a filter chain are covered,
along with the handling of invalid definitions.

"""

import io

import pytest

from . import svg2pdf, svg2png

WIDE_REGION = (
    'filterUnits="userSpaceOnUse" x="-100" y="-100" width="400" height="400"')


def render_png(filter_body, color='red', opacity=1, size=20,
               region=WIDE_REGION, rect=(0, 0, None, None), extra=''):
    """Render a single filtered rectangle and return the PIL image."""
    x, y, width, height = rect
    width = size if width is None else width
    height = size if height is None else height
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}"
                 height="{size}">
<filter id="f" {region}>{filter_body}</filter>
{extra}
<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{color}"
      fill-opacity="{opacity}" filter="url(#f)"/>
</svg>'''
    from PIL import Image
    return Image.open(io.BytesIO(svg2png(bytestring=svg.encode()))).convert(
        'RGBA')


def pixel(filter_body, color='red', opacity=1, at=(10, 10), **kwargs):
    return render_png(filter_body, color, opacity, **kwargs).getpixel(at)


@pytest.mark.parametrize('color, expected', (
    ('red', (54, 54, 54, 255)),
    ('lime', (182, 182, 182, 255)),
    ('blue', (18, 18, 18, 255)),
))
def test_saturate_grayscale(color, expected):
    """``saturate`` with a zero value computes the Rec. 709 luminance."""
    assert pixel('<feColorMatrix type="saturate" values="0"/>',
                 color) == expected


def test_saturate_default_is_identity():
    """A missing value defaults to full saturation (identity)."""
    assert pixel('<feColorMatrix type="saturate"/>',
                 'rgb(10,20,30)') == (10, 20, 30, 255)


def test_saturate_half():
    assert pixel('<feColorMatrix type="saturate" values="0.5"/>',
                 'red') == (155, 27, 27, 255)


def test_matrix_identity_default():
    """An empty ``matrix`` primitive is the identity transformation."""
    assert pixel('<feColorMatrix type="matrix"/>',
                 'rgb(10,20,30)') == (10, 20, 30, 255)


def test_matrix_swap_channels():
    matrix = ('0 0 1 0 0  0 1 0 0 0  1 0 0 0 0  0 0 0 1 0')
    assert pixel(
        f'<feColorMatrix type="matrix" values="{matrix}"/>',
        'rgb(10,20,30)') == (30, 20, 10, 255)


def test_matrix_scale_alpha():
    matrix = ('1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 0.5 0')
    assert pixel(
        f'<feColorMatrix type="matrix" values="{matrix}"/>',
        'red') == (255, 0, 0, 128)


def test_matrix_constants():
    """The fifth column adds a constant offset to every channel."""
    matrix = ('0 0 0 0 0.5  0 0 0 0 0.25  0 0 0 0 0  0 0 0 1 0')
    assert pixel(
        f'<feColorMatrix type="matrix" values="{matrix}"/>',
        'red') == (128, 64, 0, 255)


def test_matrix_clamps_results():
    """Channel results are clamped to the [0, 1] range."""
    # R = 5 * (128/255) - 2 (negative-ish, kept), G/B = 5 * (128/255) -> 1
    matrix = ('5 0 0 0 -2  0 5 0 0 0  0 0 5 0 0  0 0 0 1 0')
    result = pixel(
        f'<feColorMatrix type="matrix" values="{matrix}"/>',
        'rgb(128,128,128)')
    assert result[0] == 130
    assert result[1:] == (255, 255, 255)


def test_hue_rotate_identity():
    for body in ('<feColorMatrix type="hueRotate" values="0"/>',
                 '<feColorMatrix type="hueRotate"/>'):
        assert pixel(body, 'rgb(10,20,30)') == (10, 20, 30, 255)


def test_hue_rotate_180():
    assert pixel('<feColorMatrix type="hueRotate" values="180"/>',
                 'red') == (0, 108, 108, 255)


@pytest.mark.parametrize('unit', ('deg', 'grad', 'rad', 'turn'))
def test_hue_rotate_units(unit):
    """Angles accept deg, grad, rad and turn suffixes."""
    values = {
        'deg': '90deg', 'grad': '100grad',
        'rad': '1.5707963267948966rad', 'turn': '0.25turn'}[unit]
    assert pixel(
        f'<feColorMatrix type="hueRotate" values="{values}"/>',
        'red') == pixel(
        '<feColorMatrix type="hueRotate" values="90"/>', 'red')


def test_luminance_to_alpha():
    assert pixel('<feColorMatrix type="luminanceToAlpha"/>',
                 'red') == (0, 0, 0, 54)
    assert pixel('<feColorMatrix type="luminanceToAlpha"/>',
                 'white') == (0, 0, 0, 255)
    assert pixel('<feColorMatrix type="luminanceToAlpha"/>',
                 'black') == (0, 0, 0, 0)


def test_transparent_pixels_untouched():
    """Fully transparent pixels stay transparent for every mode."""
    for mode, values in (
            ('matrix', '1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 1 0'),
            ('saturate', '0'), ('hueRotate', '90'),
            ('luminanceToAlpha', '')):
        attr = f'values="{values}"' if values else ''
        image = render_png(
            f'<feColorMatrix type="{mode}" {attr}/>',
            rect=(4, 4, 12, 12))
        assert image.getpixel((0, 0)) == (0, 0, 0, 0)
        assert image.getpixel((19, 19)) == (0, 0, 0, 0)


def test_unpremultiplied_color_on_translucent_pixel():
    """A 50% transparent red stays red, it must not become darkened."""
    assert pixel('<feColorMatrix type="saturate"/>', 'red',
                 opacity=0.5) == (255, 0, 0, 128)


def test_luminance_to_alpha_keeps_source_alpha():
    """luminanceToAlpha scales the luminance by the source alpha."""
    assert pixel('<feColorMatrix type="luminanceToAlpha"/>', 'white',
                 opacity=0.5) == (0, 0, 0, 128)


def test_filter_combined_with_node_opacity():
    """The node opacity is applied once, after the filter has run."""
    red, green, blue, alpha = pixel(
        '<feColorMatrix type="saturate" values="0"/>', 'red', opacity=0.5)
    assert (red, green, blue) == (54, 54, 54)
    assert abs(alpha - 128) <= 1


def test_source_alpha_input():
    """``SourceAlpha`` provides the source alpha as a black color."""
    body = (
        '<feColorMatrix in="SourceAlpha" type="matrix" '
        'values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 1 0"/>')
    assert pixel(body, 'rgb(100,150,200)') == (0, 0, 0, 255)
    assert pixel(body, 'rgb(100,150,200)', opacity=0.5) == (0, 0, 0, 128)


def test_filter_on_group():
    """A filter referenced by a group applies to all its children."""
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">
<filter id="g" {WIDE_REGION}>
  <feColorMatrix type="saturate" values="0"/>
</filter>
<g filter="url(#g)">
  <rect x="0" y="0" width="20" height="20" fill="red"/>
  <rect x="20" y="20" width="20" height="20" fill="lime"/>
</g></svg>'''
    from PIL import Image
    image = Image.open(io.BytesIO(svg2png(bytestring=svg.encode()))).convert(
        'RGBA')
    assert image.getpixel((10, 10)) == (54, 54, 54, 255)
    assert image.getpixel((30, 30)) == (182, 182, 182, 255)


def test_chained_matrices_declaration_order():
    """Two primitives are combined in their declaration order."""
    body = (
        '<feColorMatrix type="saturate" values="0"/>'
        '<feColorMatrix type="matrix" '
        'values="2 0 0 0 0  0 2 0 0 0  0 0 2 0 0  0 0 0 1 0"/>')
    assert pixel(body, 'red') == (108, 108, 108, 255)


def test_named_result_input():
    """A primitive can reference a previous ``result`` by name."""
    body = (
        '<feColorMatrix type="saturate" values="0" result="gray"/>'
        '<feColorMatrix in="gray" type="matrix" '
        'values="1 0 0 0 0  0 1 0 0 0  0 0 0 0 0  0 0 0 1 0"/>')
    red, green, blue, alpha = pixel(body, 'rgb(10,20,30)')
    assert abs(red - 19) <= 1 and blue == 0 and alpha == 255


def test_explicit_source_graphic_input():
    """``in="SourceGraphic"`` bypasses the previous primitive."""
    body = (
        '<feColorMatrix type="luminanceToAlpha"/>'
        '<feColorMatrix in="SourceGraphic" type="saturate" values="0"/>')
    assert pixel(body, 'red') == (54, 54, 54, 255)


def test_offset_then_color_matrix():
    """feOffset followed by feColorMatrix moves then grays the graphic."""
    body = ('<feOffset dx="10" dy="0"/>'
            '<feColorMatrix type="saturate" values="0"/>')
    image = render_png(body, 'red', size=40, rect=(0, 10, 20, 20))
    opaque = [
        (x, y) for y in range(40) for x in range(40)
        if image.getpixel((x, y))[3]]
    assert min(x for x, _ in opaque) == 10
    assert max(x for x, _ in opaque) == 29
    assert image.getpixel((20, 20)) == (54, 54, 54, 255)


def test_blend_multiply_with_white():
    body = (
        '<feFlood flood-color="white" result="white"/>'
        '<feBlend in="SourceGraphic" in2="white" mode="multiply"/>')
    result = pixel(body, 'rgb(100,150,200)')
    assert all(abs(a - b) <= 1 for a, b in zip(
        result[:3], (100, 150, 200)))


def test_blend_screen_with_black():
    body = (
        '<feFlood flood-color="black" result="black"/>'
        '<feBlend in="SourceGraphic" in2="black" mode="screen"/>')
    result = pixel(body, 'rgb(100,150,200)')
    assert all(abs(a - b) <= 1 for a, b in zip(
        result[:3], (100, 150, 200)))


def test_unknown_named_input_is_transparent():
    body = '<feColorMatrix in="missing" type="saturate" values="0"/>'
    assert pixel(body, 'red') == (0, 0, 0, 0)


def test_filter_region_clipping():
    """The filter region clips the final result."""
    region = ('filterUnits="userSpaceOnUse" x="0" y="0" '
              'width="20" height="40"')
    image = render_png(
        '<feColorMatrix type="saturate" values="0"/>', 'red', size=40,
        region=region, rect=(0, 0, 40, 40))
    assert image.getpixel((10, 20))[3] == 255
    assert image.getpixel((30, 20))[3] == 0


def test_primitive_subregion_clipping():
    """A primitive subregion clips only that primitive's output."""
    region = ('filterUnits="userSpaceOnUse" x="0" y="0" '
              'width="40" height="40"')
    body = ('<feColorMatrix type="saturate" values="0" '
            'x="0" y="0" width="20" height="40"/>')
    image = render_png(body, 'red', size=40, region=region,
                       rect=(0, 0, 40, 40))
    assert image.getpixel((10, 20)) == (54, 54, 54, 255)
    assert image.getpixel((30, 20))[3] == 0


@pytest.mark.parametrize('body', (
    '<feColorMatrix type="bogus"/>',
    '<feColorMatrix type="matrix" values="' + '1 ' * 19 + '"/>',
    '<feColorMatrix type="matrix" values="' + '1 ' * 21 + '"/>',
    '<feColorMatrix type="saturate" values="0.5 0.5"/>',
    '<feColorMatrix type="hueRotate" values="90 90"/>',
    '<feColorMatrix type="saturate" values="NaN"/>',
    '<feColorMatrix type="hueRotate" values="NaN"/>',
    '<feColorMatrix type="matrix" values="abc"/>',
    '<feColorMatrix type="saturate" values="abc"/>',
))
def test_invalid_definition_paints_source_unchanged(body):
    """Invalid primitives never corrupt the canvas: source is painted."""
    assert pixel(body, 'red') == (255, 0, 0, 255)


def test_invalid_primitive_keeps_outside_canvas_empty():
    image = render_png(
        '<feColorMatrix type="bogus"/>', 'red', size=40,
        rect=(10, 10, 20, 20))
    assert image.getpixel((0, 0)) == (0, 0, 0, 0)
    assert image.getpixel((39, 39)) == (0, 0, 0, 0)
    assert image.getpixel((20, 20)) == (255, 0, 0, 255)


def test_missing_filter_reference_renders_unfiltered():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">'
           '<rect width="20" height="20" fill="red" filter="url(#nope)"/>'
           '</svg>')
    from PIL import Image
    image = Image.open(io.BytesIO(svg2png(bytestring=svg.encode()))).convert(
        'RGBA')
    assert image.getpixel((10, 10)) == (255, 0, 0, 255)


def test_pdf_output_embeds_filter_result():
    """The raster filter result is embedded into the PDF surface."""
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">
<filter id="g" {WIDE_REGION}>
  <feColorMatrix type="saturate" values="0"/>
</filter>
<rect x="10" y="10" width="20" height="20" fill="red" filter="url(#g)"/>
<rect x="0" y="0" width="8" height="8" fill="blue"/>
</svg>'''
    pdf = svg2pdf(bytestring=svg.encode())
    assert pdf.startswith(b'%PDF')
