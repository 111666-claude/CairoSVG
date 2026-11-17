"""
Tests for the ``feComponentTransfer`` SVG filter primitive.

The checks use tiny fixed-color images and compare the premultiplied ARGB32
output of the filter with an independent, straight-RGBA reference
implementation.

"""

import io
import math
import struct

import cairocffi as cairo
import pytest

from . import svg2png
from .filters import (
    IDENTITY, component_transfer, parse_component_transfer)
from .parser import Tree

PIXEL_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" width="{n}" height="1"
     shape-rendering="crispEdges">
  <defs><filter id="f" filterUnits="userSpaceOnUse" x="0" y="0"
          width="{n}" height="1">
    <feComponentTransfer>{body}</feComponentTransfer>
  </filter></defs>
  <g filter="url(#f)">{rects}</g>
</svg>"""

FUNCS = {
    'identity': (
        '<feFuncR type="identity"/><feFuncG type="identity"/>'
        '<feFuncB type="identity"/><feFuncA type="identity"/>'),
    'linear': (
        '<feFuncR type="linear" slope="0.5" intercept="0.25"/>'
        '<feFuncG type="linear" slope="2" intercept="-0.5"/>'
        '<feFuncB type="linear" slope="-1" intercept="1"/>'
        '<feFuncA type="linear" slope="0.5" intercept="0.1"/>'),
    'gamma': (
        '<feFuncR type="gamma" amplitude="1" exponent="2" offset="0"/>'
        '<feFuncG type="gamma" amplitude="0.8" exponent="0.5" offset="0.1"/>'
        '<feFuncB type="gamma" amplitude="1.2" exponent="3" offset="-0.2"/>'
        '<feFuncA type="gamma" amplitude="1" exponent="1.5" offset="0"/>'),
    'table': (
        '<feFuncR type="table" tableValues="0 0.25 0.75 1"/>'
        '<feFuncG type="table" tableValues="1 0.5 0"/>'
        '<feFuncB type="table" tableValues="0.1 0.9"/>'
        '<feFuncA type="table" tableValues="0 0.5 1"/>'),
    'discrete': (
        '<feFuncR type="discrete" tableValues="0 0.33 0.66 1"/>'
        '<feFuncG type="discrete" tableValues="1 0.5 0"/>'
        '<feFuncB type="discrete" tableValues="0.2 0.8"/>'
        '<feFuncA type="discrete" tableValues="0 0.5 1"/>'),
}

COLORS = ['ff0000', '00ff00', '0000ff', '808080', 'ffffff', '000000',
          '336699', 'cc9933']
OPACITIES = [1.0, 0.5, 0.25, 0.75, 0.0, 0.1, 0.9, 1.0]


def _parse(body):
    tree = Tree(bytestring=(
        f'<svg xmlns="http://www.w3.org/2000/svg">'
        f'<filter id="f"><feComponentTransfer>{body}'
        f'</feComponentTransfer></filter></svg>').encode())
    return parse_component_transfer(tree.children[0].children[0])


def _reference_word(color, opacity, body):
    """Independent straight-RGBA reference, returning the ARGB32 word."""
    straight = (
        [int(color[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [opacity])
    transfers = _parse(body)
    mapped = []
    for index, value in enumerate(straight):
        transfer = transfers[index]
        kind = transfer[0]
        if kind == 'linear':
            value = transfer[1] * value + transfer[2]
        elif kind == 'gamma':
            _, amplitude, exponent, offset = transfer
            value = amplitude * value ** exponent + offset
        elif kind in ('table', 'discrete'):
            table = transfer[1]
            count = len(table)
            if kind == 'table':
                position = value * (count - 1)
                lower = int(position)
                value = (
                    table[-1] if lower >= count - 1 else
                    table[lower] +
                    (position - lower) * (table[lower + 1] - table[lower]))
            else:
                value = table[min(int(value * count), count - 1)]
        if not math.isfinite(value):
            value = 0.0
        mapped.append(min(1.0, max(0.0, value)))

    red, green, blue, alpha = mapped
    if alpha == 0:
        return 0
    premultiplied = (round(channel * alpha * 255)
                     for channel in (red, green, blue))
    pr, pg, pb = premultiplied
    return (round(alpha * 255) << 24) | (pr << 16) | (pg << 8) | pb


def _render_words(body, colors, opacities):
    rects = ''.join(
        f'<rect x="{i}" y="0" width="1" height="1" fill="#{color}" '
        f'fill-opacity="{opacity}"/>'
        for i, (color, opacity) in enumerate(zip(colors, opacities)))
    svg = PIXEL_SVG.format(n=len(colors), body=body, rects=rects).encode()
    surface = cairo.ImageSurface.create_from_png(
        io.BytesIO(svg2png(bytestring=svg)))
    count = len(colors)
    raw = bytes(surface.get_data())[:count * 4]
    # Native 32-bit ARGB32 words, one per pixel.
    return list(struct.unpack(f'{count}I', raw))


@pytest.mark.parametrize('func', list(FUNCS))
def test_component_transfer_matches_reference(func):
    body = FUNCS[func]
    words = _render_words(body, COLORS, OPACITIES)
    for index, (color, opacity) in enumerate(zip(COLORS, OPACITIES)):
        if opacity == 0:
            # The straight color of a transparent pixel is undefined; only
            # the mapped alpha channel is observable there.
            assert abs(((words[index] >> 24) & 255) -
                       ((_reference_word(color, opacity, body) >> 24) & 255)
                       ) <= 1
        else:
            expected = _reference_word(color, opacity, body)
            for shift in (0, 8, 16, 24):
                assert abs(((words[index] >> shift) & 255) -
                           ((expected >> shift) & 255)) <= 1


def test_transparent_pixel_stays_transparent():
    body = FUNCS['gamma']
    words = _render_words(body, ['ffffff'], [0.0])
    assert words == [0]


def test_unspecified_channels_are_unchanged():
    body = ('<feFuncR type="linear" slope="0" intercept="1"/>'
            '<feFuncB type="gamma" amplitude="1" exponent="2" offset="0"/>')
    words = _render_words(body, ['123456'], [1.0])[0]
    assert (words >> 16) & 255 == 255                      # R forced to 1
    assert (words >> 8) & 255 == 0x34                      # G untouched
    assert words & 255 == round((0x56 / 255) ** 2 * 255)   # B gamma mapped


def test_empty_table_is_identity():
    assert _parse('<feFuncR type="table" tableValues=""/>')[0] is IDENTITY
    assert _parse(
        '<feFuncB type="discrete" tableValues="  "/>')[2] is IDENTITY


@pytest.mark.parametrize('exponent', ('0', '-3', 'nan', 'inf', '-inf'))
def test_invalid_gamma_exponent_is_identity(exponent):
    body = (f'<feFuncR type="gamma" amplitude="1" exponent="{exponent}" '
            'offset="0"/>')
    assert _parse(body)[0] is IDENTITY


@pytest.mark.parametrize('body', (
    '<feFuncR type="linear" slope="inf" intercept="0"/>',
    '<feFuncG type="linear" slope="1" intercept="nan"/>',
    '<feFuncB type="gamma" amplitude="nan" exponent="2" offset="0"/>',
    '<feFuncB type="table" tableValues="0 nan 1"/>',
    '<feFuncA type="discrete" tableValues="0 0.5 inf"/>',
))
def test_non_finite_parameters_are_identity(body):
    transfers = _parse(body)
    assert all(transfer is IDENTITY for transfer in transfers.values())


def test_repeated_channel_uses_last_definition():
    body = ('<feFuncR type="linear" slope="0" intercept="0"/>'
            '<feFuncR type="linear" slope="2" intercept="1"/>')
    assert _parse(body)[0] == ('linear', 2.0, 1.0)


def test_unknown_type_and_missing_func_are_identity():
    transfers = _parse('<feFuncG type="bogus"/>')
    assert all(transfer is IDENTITY for transfer in transfers.values())
    assert _parse('<feFuncB/>')[2] is IDENTITY


def test_uppercase_tags_are_recognized():
    # The parser preserves the case of filter primitive tags, so matching must
    # be case-insensitive for both the primitive and its channel functions.
    body = '<FUNCA type="linear" slope="0.5" intercept="0"/>'
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1">'
        '<defs><filter id="f"><FECOMPONENTTRANSFER>'
        f'{body}</FECOMPONENTTRANSFER></filter></defs>'
        '<g filter="url(#f)"><rect width="1" height="1" fill="red"/></g>'
        '</svg>').encode()
    surface = cairo.ImageSurface.create_from_png(
        io.BytesIO(svg2png(bytestring=svg)))
    word = struct.unpack('I', bytes(surface.get_data())[:4])[0]
    # Opaque input with alpha mapped through 0.5 -> alpha ~= 128. If the
    # uppercase tags were ignored, alpha would stay 255.
    assert abs(((word >> 24) & 255) - 128) <= 2


def test_oversized_inputs_are_finite_and_clamped():
    from .filters import _clamp, _lookup
    # Pathological gamma parameters overflow a float but must stay finite and
    # be clamped to the legal [0, 1] range instead of propagating NaN/inf.
    assert _clamp(1e308 * 1.0 ** 1000) == 1.0
    assert _clamp(float('inf')) == 0.0
    assert _clamp(float('nan')) == 0.0
    big_table = [i / 1000 for i in range(1001)]
    assert 0.0 <= _lookup(0.7, ('table', big_table)) <= 1.0
    assert 0.0 <= _lookup(1.0, ('discrete', big_table)) <= 1.0
    words = _render_words(
        '<feFuncR type="table" tableValues="'
        + ' '.join(str(i / 1000) for i in range(1001)) + '"/>',
        ['7f3f10'], [1.0])
    assert 0 <= words[0] <= 0xFFFFFFFF


def test_source_surface_is_not_modified():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 1)
    buffer = surface.get_data()
    # Native ARGB32 words: opaque red and 50% gray.
    buffer[:8] = struct.pack('II', 0xFFFF0000, 0x80808080)
    surface.mark_dirty()
    before = bytes(surface.get_data())
    node = Tree(bytestring=(
        b'<svg xmlns="http://www.w3.org/2000/svg"><filter id="f">'
        b'<feComponentTransfer><feFuncR type="linear" slope="0" '
        b'intercept="1"/></feComponentTransfer></filter></svg>'
    )).children[0].children[0]
    result = component_transfer(node, surface)
    assert result is not surface
    assert bytes(surface.get_data()) == before
