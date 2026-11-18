"""
Tests for the raster filter pipeline, mainly ``feComposite``.

The expected pixels are computed by a small, independent implementation of
the SVG 1.1 compositing formulas (Porter-Duff and arithmetic) written directly
from the specification.  This avoids just re-testing the implementation with
itself: the reference below uses plain Python floats on premultiplied colors
and does not import the production code.

"""

import io
import warnings

import numpy as np
import pytest

from . import svg2png

# Two fixed semi-transparent layers, rendered as feFlood primitives so that
# the test does not depend on shape rasterization for its input pixels.
# A is red with alpha 0.6, B is blue with alpha 0.4; they overlap on 20x20.
LAYER_SVG = '''\
<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40"
     viewBox="0 0 40 40">
  <defs>
    <filter id="f" x="0" y="0" width="40" height="40"
            filterUnits="userSpaceOnUse">
      <feFlood flood-color="#ff0000" flood-opacity="0.6"
               x="5" y="10" width="20" height="20" result="A"/>
      <feFlood flood-color="#0000ff" flood-opacity="0.4"
               x="15" y="10" width="20" height="20" result="B"/>
      {primitive}
    </filter>
  </defs>
  <rect width="40" height="40" fill="none" filter="url(#f)"/>
</svg>
'''

# Premultiplied RGBA bytes actually stored by Cairo for each feFlood layer.
# red #ff0000 at 0.6 -> premultiplied R = round(255 * 0.6) = 153, A = 153.
A_BYTES = (153, 0, 0, 153)
# blue #0000ff at 0.4 -> premultiplied B = round(255 * 0.4) = 102, A = 102.
B_BYTES = (0, 0, 102, 102)
TRANSPARENT = (0, 0, 0, 0)
# Unpremultiplied 8-bit RGBA of layer A as stored in the PNG output.
A_RGBA = (255, 0, 0, 153)


def _png_pixels(svg_source):
    """Render an SVG string and return its pixels as an RGBA uint8 array."""
    png = svg2png(bytestring=svg_source.encode('ascii'))
    from PIL import Image

    # A fully opaque output is written as RGB; normalize to RGBA so that
    # every render has the same four-channel layout.
    return np.array(
        Image.open(io.BytesIO(png)).convert('RGBA'))


def _round_channel(value):
    """Clamp and round a normalized channel to an 8-bit value."""
    return min(255, max(0, int(round(value))))


def _unpremultiply(premult_bytes):
    """Reverse Cairo/pixman's premultiplied storage.

    pixman unpremultiplies on integer bytes with
    ``(channel * 255 + alpha // 2) // alpha`` (see pixman's
    unpremultiply table), not a rounded float division.

    """
    alpha = premult_bytes[3]
    if alpha == 0:
        return (0, 0, 0, 0)
    return tuple(
        min(255, (channel * 255 + alpha // 2) // alpha)
        for channel in premult_bytes[:3]) + (alpha,)


def _expected(s_bytes, d_bytes, operator, ks=(0, 0, 0, 0)):
    """Reference composite, returned as 8-bit unpremultiplied RGBA.

    Inputs are the premultiplied RGBA bytes Cairo actually stores, so the
    reference is affected by the same quantization as the production pipeline.

    """
    s = tuple(channel / 255 for channel in s_bytes)
    d = tuple(channel / 255 for channel in d_bytes)
    premultiplied = _reference_composite(s, d, operator, ks)
    premult_bytes = tuple(
        _round_channel(channel * 255) for channel in premultiplied)
    return _unpremultiply(premult_bytes)


# ---------------------------------------------------------------------------
# Independent reference implementation of the SVG feComposite formulas
# ---------------------------------------------------------------------------

def _reference_composite(s, d, operator, ks=(0, 0, 0, 0)):
    """Composite premultiplied normalized RGBA tuples ``s`` and ``d``."""
    sr, sg, sb, sa = s
    dr, dg, db, da = d
    if operator == 'over':
        out = (sr + dr * (1 - sa), sg + dg * (1 - sa),
               sb + db * (1 - sa), sa + da * (1 - sa))
    elif operator == 'in':
        out = (sr * da, sg * da, sb * da, sa * da)
    elif operator == 'out':
        out = (sr * (1 - da), sg * (1 - da), sb * (1 - da), sa * (1 - da))
    elif operator == 'atop':
        out = (sr * da + dr * (1 - sa),
               sg * da + dg * (1 - sa),
               sb * da + db * (1 - sa),
               sa * da + da * (1 - sa))
    elif operator == 'xor':
        out = (sr * (1 - da) + dr * (1 - sa),
               sg * (1 - da) + dg * (1 - sa),
               sb * (1 - da) + db * (1 - sa),
               sa * (1 - da) + da * (1 - sa))
    else:  # arithmetic
        k1, k2, k3, k4 = ks
        out = tuple(
            min(1, max(0, k1 * sc * dc + k2 * sc + k3 * dc + k4))
            for sc, dc in zip(s, d))
    return tuple(float(channel) for channel in out)


OPERATORS = ('over', 'in', 'out', 'atop', 'xor')

# Probe points: A-only, overlap, B-only and a point outside both layers.
PROBES = {
    'a_only': (8, 20),
    'overlap': (20, 20),
    'b_only': (30, 20),
    'outside': (2, 2),
}


@pytest.mark.parametrize('operator', OPERATORS)
def test_porter_duff_operators_pixels(operator):
    """Check every Porter-Duff operator against the spec formulas."""
    primitive = (
        f'<feComposite in="A" in2="B" operator="{operator}"/>')
    pixels = _png_pixels(LAYER_SVG.format(primitive=primitive))

    expected = {
        'a_only': _expected(A_BYTES, TRANSPARENT, operator),
        'overlap': _expected(A_BYTES, B_BYTES, operator),
        'b_only': _expected(TRANSPARENT, B_BYTES, operator),
        'outside': (0, 0, 0, 0),
    }
    for name, (x, y) in PROBES.items():
        assert tuple(pixels[y, x]) == expected[name], (
            f'{operator}: {name} gives {tuple(pixels[y, x])}, '
            f'expected {expected[name]}')


def test_arithmetic_operator_pixels():
    """Check arithmetic with non-trivial k1-k4 coefficients."""
    ks = (0.2, 0.3, 0.5, 0.05)
    primitive = (
        '<feComposite in="A" in2="B" operator="arithmetic" '
        'k1="0.2" k2="0.3" k3="0.5" k4="0.05"/>')
    pixels = _png_pixels(LAYER_SVG.format(primitive=primitive))

    expected = {
        'a_only': _expected(
            A_BYTES, TRANSPARENT, 'arithmetic', ks),
        'overlap': _expected(A_BYTES, B_BYTES, 'arithmetic', ks),
        'b_only': _expected(
            TRANSPARENT, B_BYTES, 'arithmetic', ks),
        'outside': _expected(
            TRANSPARENT, TRANSPARENT, 'arithmetic', ks),
    }
    for name, (x, y) in PROBES.items():
        assert tuple(pixels[y, x]) == expected[name]


def test_arithmetic_coefficients_are_clipped():
    """Out-of-range arithmetic results are clipped, including alpha."""
    # k4 = 1 forces every channel, including alpha, to clamp at 255 even
    # where both inputs are fully transparent.
    primitive = (
        '<feComposite in="A" in2="B" operator="arithmetic" '
        'k1="0" k2="0" k3="0" k4="1"/>')
    pixels = _png_pixels(LAYER_SVG.format(primitive=primitive))
    assert tuple(pixels[2, 2]) == (255, 255, 255, 255)

    # Negative k4 clamps to 0: the whole image stays transparent.
    primitive = (
        '<feComposite in="A" in2="B" operator="arithmetic" '
        'k1="0" k2="0" k3="0" k4="-2"/>')
    pixels = _png_pixels(LAYER_SVG.format(primitive=primitive))
    assert not pixels[:, :, 3].any()


def test_arithmetic_default_coefficients():
    """Missing k attributes default to 0, giving a transparent result."""
    primitive = (
        '<feComposite in="A" in2="B" operator="arithmetic"/>')
    pixels = _png_pixels(LAYER_SVG.format(primitive=primitive))
    assert not pixels[:, :, 3].any()


def test_non_finite_coefficients():
    """NaN and infinite coefficients are replaced by 0 with a warning."""
    for bad_value in ('NaN', 'Infinity', '-Infinity'):
        primitive = (
            '<feComposite in="A" in2="B" operator="arithmetic" '
            f'k1="0" k2="1" k3="0" k4="{bad_value}"/>')
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            pixels = _png_pixels(LAYER_SVG.format(primitive=primitive))
        assert any('not a finite number' in str(w.message)
                   for w in recorded)
        # k2 = 1 reproduces the source layer, the bad k4 contributes nothing.
        x, y = PROBES['a_only']
        assert tuple(pixels[y, x]) == A_RGBA
        assert np.isfinite(pixels.astype(float)).all()


def test_unparsable_coefficients_become_zero():
    """A coefficient that is not a number at all falls back to 0."""
    primitive = (
        '<feComposite in="A" in2="B" operator="arithmetic" '
        'k1="abc" k2="1" k3="" k4="0"/>')
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        pixels = _png_pixels(LAYER_SVG.format(primitive=primitive))
    # k1 and k3 are invalid and each triggers a warning.
    bad = [w for w in recorded if 'not a finite number' in str(w.message)]
    assert len(bad) == 2
    assert np.isfinite(pixels.astype(float)).all()
    # With k2 = 1 and the other coefficients at 0, the source layer survives.
    x, y = PROBES['a_only']
    assert tuple(pixels[y, x]) == A_RGBA


def test_unknown_operator_falls_back_to_over():
    """An unknown operator behaves as over and emits a warning."""
    primitive = (
        '<feComposite in="A" in2="B" operator="lighter"/>')
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        pixels = _png_pixels(LAYER_SVG.format(primitive=primitive))
    assert any('Unknown feComposite operator' in str(w.message)
               for w in recorded)
    x, y = PROBES['overlap']
    assert tuple(pixels[y, x]) == _expected(
        A_BYTES, B_BYTES, 'over')


def test_missing_inputs_are_transparent():
    """Unknown and unsupported input references resolve to transparent."""
    # in2 references a missing result: composite with transparent black.
    primitive = (
        '<feComposite in="A" in2="missing" operator="over"/>')
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        pixels = _png_pixels(LAYER_SVG.format(primitive=primitive))
    assert any('does not reference any previous result' in str(w.message)
               for w in recorded)
    x, y = PROBES['a_only']
    assert tuple(pixels[y, x]) == A_RGBA

    # BackgroundImage is not captured by CairoSVG: transparent input.
    primitive = (
        '<feComposite in="A" in2="BackgroundImage" operator="in"/>')
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        pixels = _png_pixels(LAYER_SVG.format(primitive=primitive))
    assert any('not supported' in str(w.message) for w in recorded)
    assert not pixels[:, :, 3].any()


def test_source_graphic_and_source_alpha():
    """SourceGraphic and SourceAlpha resolve the painted group correctly."""
    svg = '''\
<svg xmlns="http://www.w3.org/2000/svg" width="30" height="30"
     viewBox="0 0 30 30">
  <defs>
    <filter id="f" x="0" y="0" width="30" height="30"
            filterUnits="userSpaceOnUse">
      <feComposite in="SourceAlpha" in2="SourceGraphic" operator="in"/>
    </filter>
  </defs>
  <rect x="5" y="5" width="20" height="20" fill="#ff8800"
        fill-opacity="0.5" filter="url(#f)"/>
</svg>
'''
    pixels = _png_pixels(svg)
    # in(SourceAlpha, SourceGraphic) = SourceAlpha * a(SourceGraphic).
    # SourceAlpha is (0, 0, 0, 0.5); multiplied by 0.5 -> alpha 0.25, black.
    assert tuple(pixels[15, 15]) == (0, 0, 0, round(255 * 0.25))
    # Outside the graphic everything stays transparent.
    assert tuple(pixels[1, 1]) == (0, 0, 0, 0)


def test_transparent_pixels_have_no_fringe():
    """Fully transparent pixels keep black RGB (premultiplied storage).

    The 'out' operator removes the overlap region; it must not leave colored
    fringe pixels where alpha is zero.

    """
    primitive = '<feComposite in="A" in2="B" operator="out"/>'
    pixels = _png_pixels(LAYER_SVG.format(primitive=primitive))
    transparent = pixels[pixels[:, :, 3] == 0]
    assert (transparent[:, :3] == 0).all()


def test_filter_region_clips_result():
    """The filter output is clipped to the declared filter region."""
    svg = '''\
<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40"
     viewBox="0 0 40 40">
  <defs>
    <filter id="f" x="10" y="10" width="15" height="15"
            filterUnits="userSpaceOnUse">
      <feFlood flood-color="red" flood-opacity="1" result="r"/>
      <!-- feComposite routes the filter through the raster pipeline; over
           with a transparent in2 keeps the flood result unchanged. -->
      <feComposite in="r" in2="SourceGraphic" operator="over"/>
    </filter>
  </defs>
  <rect width="40" height="40" fill="none" filter="url(#f)"/>
</svg>
'''
    pixels = _png_pixels(svg)
    # Inside the region: opaque red.
    assert tuple(pixels[17, 17]) == (255, 0, 0, 255)
    # Outside: untouched, transparent.
    assert tuple(pixels[5, 5]) == (0, 0, 0, 0)
    assert tuple(pixels[30, 30]) == (0, 0, 0, 0)


def test_primitive_subregion_clips_result():
    """A primitive x/y/width/height clips its own output."""
    svg = '''\
<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40"
     viewBox="0 0 40 40">
  <defs>
    <filter id="f" x="0" y="0" width="40" height="40"
            filterUnits="userSpaceOnUse">
      <feFlood flood-color="red" flood-opacity="1"
               x="8" y="8" width="10" height="10" result="r"/>
      <feComposite in="r" in2="SourceGraphic" operator="over"/>
    </filter>
  </defs>
  <rect width="40" height="40" fill="none" filter="url(#f)"/>
</svg>
'''
    pixels = _png_pixels(svg)
    assert tuple(pixels[13, 13]) == (255, 0, 0, 255)
    assert tuple(pixels[5, 5]) == (0, 0, 0, 0)
    assert tuple(pixels[25, 25]) == (0, 0, 0, 0)


def test_chained_primitives_use_previous_result():
    """A primitive without 'in' consumes the previous primitive result."""
    svg = '''\
<svg xmlns="http://www.w3.org/2000/svg" width="30" height="30"
     viewBox="0 0 30 30">
  <defs>
    <filter id="f" x="0" y="0" width="30" height="30"
            filterUnits="userSpaceOnUse">
      <feFlood flood-color="red" flood-opacity="1"
               x="5" y="5" width="20" height="20" result="r"/>
      <!-- No 'in': 'over' composites the previous flood result with an
           empty SourceGraphic, so the flood survives unchanged. -->
      <feComposite in2="SourceGraphic" operator="over"/>
    </filter>
  </defs>
  <rect width="30" height="30" fill="none" filter="url(#f)"/>
</svg>
'''
    # over(source=flood, in2=empty SourceGraphic) keeps the flood visible.
    pixels = _png_pixels(svg)
    assert pixels[:, :, 3].any()
    assert tuple(pixels[15, 15]) == (255, 0, 0, 255)
    assert tuple(pixels[2, 2]) == (0, 0, 0, 0)


def test_filter_aligns_under_viewbox_scale():
    """The raster result stays aligned when the CTM scales user space."""
    svg = '''\
<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80"
     viewBox="0 0 40 40">
  <defs>
    <filter id="f" x="0" y="0" width="40" height="40"
            filterUnits="userSpaceOnUse">
      <feFlood flood-color="red" flood-opacity="1"
               x="5" y="5" width="10" height="10" result="r"/>
      <feComposite in="r" in2="SourceGraphic" operator="over"/>
    </filter>
  </defs>
  <rect width="40" height="40" fill="none" filter="url(#f)"/>
</svg>
'''
    # User rect (5,5,10,10) maps to device (10,10,20,20) at a 2x scale.
    pixels = _png_pixels(svg)
    assert tuple(pixels[15, 15]) == (255, 0, 0, 255)
    assert tuple(pixels[5, 5]) == (0, 0, 0, 0)
    assert tuple(pixels[35, 35]) == (0, 0, 0, 0)


def test_named_result_chain_and_offset():
    """feOffset output can be fed to feComposite through a result name."""
    svg = '''\
<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40"
     viewBox="0 0 40 40">
  <defs>
    <filter id="f" x="0" y="0" width="40" height="40"
            filterUnits="userSpaceOnUse">
      <feOffset in="SourceGraphic" dx="10" dy="0" result="shifted"/>
      <feComposite in="shifted" in2="SourceGraphic" operator="xor"/>
    </filter>
  </defs>
  <rect x="5" y="10" width="10" height="20" fill="#00ff00"
        filter="url(#f)"/>
</svg>
'''
    pixels = _png_pixels(svg)
    # Original square location and shifted location are both visible (xor).
    assert pixels[20, 7, 3] > 0
    assert pixels[20, 20, 3] > 0
    # The point beyond both squares is transparent.
    assert tuple(pixels[20, 30]) == (0, 0, 0, 0)


def test_nested_filter_on_group_children():
    """A filter applied to an inner group works inside a filtered parent."""
    svg = '''\
<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40"
     viewBox="0 0 40 40">
  <defs>
    <filter id="outer" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="0"/>
      <feComposite in="SourceGraphic" in2="SourceAlpha" operator="in"/>
    </filter>
    <filter id="inner" x="0" y="0" width="40" height="40"
            filterUnits="userSpaceOnUse">
      <feComposite in="SourceGraphic" in2="SourceGraphic" operator="over"/>
    </filter>
  </defs>
  <g filter="url(#outer)">
    <rect x="10" y="10" width="20" height="20" fill="#336699"
          filter="url(#inner)"/>
  </g>
</svg>
'''
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        pixels = _png_pixels(svg)
    # in(SourceGraphic, SourceAlpha): colors survive, alpha multiplied once.
    assert pixels[20, 20, 3] > 0
    assert tuple(pixels[2, 2]) == (0, 0, 0, 0)


def test_arithmetic_identity_coefficients():
    """k2 = 1 (others 0) makes arithmetic return the source image."""
    primitive = (
        '<feComposite in="A" in2="B" operator="arithmetic" '
        'k1="0" k2="1" k3="0" k4="0"/>')
    pixels = _png_pixels(LAYER_SVG.format(primitive=primitive))
    x, y = PROBES['a_only']
    assert tuple(pixels[y, x]) == A_RGBA
    # Where A is transparent, the source pixels are transparent too.
    assert tuple(pixels[PROBES['b_only'][1], PROBES['b_only'][0]]) == (
        0, 0, 0, 0)


def test_render_returns_png_bytes():
    """The pipeline output is a valid PNG with the requested dimensions."""
    png = svg2png(bytestring=LAYER_SVG.format(
        primitive='<feComposite in="A" in2="B" operator="over"/>').encode())
    assert png[:8] == b'\211PNG\r\n\032\n'
    from PIL import Image
    assert Image.open(io.BytesIO(png)).size == (40, 40)


def test_formulas_match_cairo_native_operators():
    """Cross-check the NumPy formulas against Cairo's own Porter-Duff ops.

    Cairo implements the same W3C operators independently; composing the two
    layers with its native operators must give identical premultiplied bytes.

    """
    import cairocffi as cairo

    from .filters import _composite, _new_image

    cairo_operators = {
        'over': cairo.OPERATOR_OVER,
        'in': cairo.OPERATOR_IN,
        'out': cairo.OPERATOR_OUT,
        'atop': cairo.OPERATOR_ATOP,
        'xor': cairo.OPERATOR_XOR,
    }

    def make_image(rgba):
        image = _new_image(1, 1)
        image.write_pixels(np.array([[list(rgba)]], dtype=np.uint8))
        return image

    source_img = make_image(A_BYTES)
    dest_img = make_image(B_BYTES)

    for operator, cairo_operator in cairo_operators.items():
        # Reference: let Cairo do the compositing on its own surfaces.
        reference = _new_image(1, 1)
        reference_context = cairo.Context(reference.surface)
        reference_context.set_source_surface(dest_img.surface)
        reference_context.paint()
        reference_context.set_operator(cairo_operator)
        reference_context.set_source_surface(source_img.surface)
        reference_context.paint()

        result = _composite(
            source_img, dest_img, operator, (0, 0, 0, 0))
        # Both store premultiplied ARGB words, so the words must match.
        reference_word = np.frombuffer(
            reference.surface.get_data(), np.uint32)[0]
        result_word = np.frombuffer(
            result.surface.get_data(), np.uint32)[0]
        assert int(result_word) == int(reference_word), operator
