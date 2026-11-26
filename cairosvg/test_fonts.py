"""
Tests for the CSS ``@font-face`` support.

These tests use a TTF installed on the system, embedded as a data URL so that
they work without any file access (and therefore with the default safe URL
fetcher). When no suitable font is available they are skipped.

"""

import base64
import io
import os

import cairocffi as cairo
import pytest

from . import parser, surface, svg2png
from .fonts import FontLibrary, _load_backend

# DejaVu Sans Mono is used because it is visually different from the default
# sans-serif font, which makes a missing @font-face rule observable.
FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
FONT_REAL_FAMILY = 'DejaVu Sans Mono'
FONT_ALIAS = 'CairoSVG Test Alias Family'
TEST_STRING = 'The quick brown fox, 0123456789!'

requires_font = pytest.mark.skipif(
    not os.path.exists(FONT_PATH) or not _load_backend(),
    reason='DejaVu Sans Mono TTF or a FreeType-enabled cairo is required')


def font_bytes():
    with open(FONT_PATH, 'rb') as fd:
        return fd.read()


def svg(*, rule=True, family=f'"{FONT_ALIAS}"', src=None, fmt='truetype'):
    font_b64 = base64.b64encode(font_bytes()).decode()
    src = src or f'data:font/ttf;base64,{font_b64}'
    if rule:
        font_face = (
            f'@font-face {{ font-family: "{FONT_ALIAS}"; '
            f'src: url({src}) format("{fmt}"); }}')
    else:
        font_face = '/* no @font-face rule */'
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="500" height="120"
     version="1.1">
  <style><![CDATA[
    {font_face}
    text {{ font-family: {family}; text-rendering: geometricPrecision; }}
  ]]></style>
  <rect width="100%" height="100%" fill="white"/>
  <text x="20" y="70" font-size="28">{TEST_STRING}</text>
</svg>'''


def render_png(document):
    return svg2png(document.encode())


def text_width(document):
    tree = parser.Tree(bytestring=document.encode())
    png_surface = surface.PNGSurface(tree, None, 96)

    def find_text(node):
        if node.tag == 'text':
            return node
        for child in node.children:
            found = find_text(child)
            if found is not None:
                return found
        return None

    node = find_text(tree)
    from .text import text as draw_text
    draw_text(png_surface, node)
    width = png_surface.context.text_extents(node.text)[4]
    png_surface.finish()
    return width


def raster(png):
    # Keep the surface alive: cairocffi reuses the memory backing get_data(),
    # so reading after the surface is garbage collected gives stale bytes.
    image_surface = cairo.ImageSurface.create_from_png(io.BytesIO(png))
    return image_surface, bytes(image_surface.get_data())


def assert_same_raster(left_png, right_png):
    _, left_raster = raster(left_png)
    _, right_raster = raster(right_png)
    assert left_raster == right_raster


@requires_font
def test_alias_matches_real_system_family():
    """An embedded TTF under an alias renders like its real system family."""
    alias_png = render_png(svg(rule=True))
    real_png = render_png(svg(rule=False, family=f'"{FONT_REAL_FAMILY}"'))
    # Same outline width and identical pixels (same font, same options).
    assert text_width(svg(rule=True)) == text_width(
        svg(rule=False, family=f'"{FONT_REAL_FAMILY}"'))
    assert_same_raster(alias_png, real_png)


@requires_font
def test_missing_rule_falls_back_to_system_font():
    """Without the rule the unknown family uses the system default font."""
    embedded = render_png(svg(rule=True))
    fallback = render_png(svg(rule=False))
    assert fallback != embedded


@requires_font
def test_font_reused_within_conversion():
    """Several text nodes share a single loaded font and render correctly."""
    document = svg(rule=True).replace(
        '<text x="20" y="70"',
        '<text x="20" y="40">AAA</text><text x="20" y="70"')
    png = render_png(document)
    assert png.startswith(b'\x89PNG')
    # Repeating the conversion gives a stable result.
    assert render_png(svg(rule=True)) == render_png(svg(rule=True))


@requires_font
def test_alias_does_not_leak_between_conversions():
    """An alias defined in one document is unknown in the next one."""
    embedded = render_png(svg(rule=True))
    fallback_first = render_png(svg(rule=False))
    fallback_second = render_png(svg(rule=False))
    assert embedded != fallback_first
    assert fallback_first == fallback_second


@requires_font
def test_safe_fetcher_blocks_external_font(tmp_path):
    """Without ``unsafe`` a relative font URL is not read from disk."""
    font_target = tmp_path / 'font.ttf'
    font_target.write_bytes(font_bytes())
    document = tmp_path / 'document.svg'
    document.write_text(svg(rule=True, src='font.ttf'))
    embedded = render_png(svg(rule=True))
    # Safe mode: the font cannot be fetched, rendering still succeeds and
    # falls back to the system font instead of raising.
    safe_png = svg2png(url=str(document))
    assert safe_png != embedded


@requires_font
def test_unsafe_fetcher_loads_relative_font(tmp_path):
    """With ``unsafe=True`` a relative TTF URL is honored like the data URL."""
    font_target = tmp_path / 'font.ttf'
    font_target.write_bytes(font_bytes())
    document = tmp_path / 'document.svg'
    document.write_text(svg(rule=True, src='font.ttf'))
    unsafe_png = svg2png(url=str(document), unsafe=True)
    assert unsafe_png == render_png(svg(rule=True))


@requires_font
def test_custom_url_fetcher_is_used():
    """A trusted custom URL fetcher provides the font bytes."""
    def fetcher(url, resource_type):
        if url.endswith('custom-font.ttf'):
            return font_bytes()
        raise OSError('blocked')

    tree = parser.Tree(
        bytestring=svg(
            rule=True, src='https://example.test/custom-font.ttf').encode(),
        url_fetcher=fetcher)
    output = io.BytesIO()
    surface.PNGSurface(tree, output, 96).finish()
    assert output.getvalue() == render_png(svg(rule=True))


@requires_font
def test_unsupported_format_is_skipped():
    """A WOFF2 source is skipped and the text falls back gracefully."""
    fallback = render_png(svg(rule=False))
    document = svg(rule=True, src='font.woff2', fmt='woff2')
    assert render_png(document) == fallback


@requires_font
def test_corrupt_font_falls_back():
    """Invalid font bytes do not crash and trigger the system fallback."""
    fallback = render_png(svg(rule=False))
    document = svg(
        rule=True, src='data:font/ttf;base64,AAAAAAAAAAAAAAAA')
    assert render_png(document) == fallback


@requires_font
def test_weight_and_style_descriptors_are_matched():
    """font-weight and font-style descriptors map to cairo constants."""
    import tinycss2

    from .surface import cairo as cairo_constants

    class _Tree:
        url = None

    def rule(css):
        at_rule = tinycss2.parse_stylesheet(css)[0]
        from .fonts import _parse_font_face_rule
        return _parse_font_face_rule(
            _Tree(), at_rule, cairo_constants)

    base = 'src: url(data:font/ttf;base64,AA);'
    assert rule(
        f'@font-face {{ font-family: F; font-weight: bold; {base} }}'
    ).weight == cairo_constants.FONT_WEIGHT_BOLD
    assert rule(
        f'@font-face {{ font-family: F; font-weight: 700; {base} }}'
    ).weight == cairo_constants.FONT_WEIGHT_BOLD
    assert rule(
        f'@font-face {{ font-family: F; font-weight: 400; {base} }}'
    ).weight == cairo_constants.FONT_WEIGHT_NORMAL
    assert rule(
        f'@font-face {{ font-family: F; font-style: italic; {base} }}'
    ).slant == cairo_constants.FONT_SLANT_ITALIC
    assert rule(
        f'@font-face {{ font-family: F; {base} }}'
    ).slant == cairo_constants.FONT_SLANT_NORMAL


@requires_font
def test_font_url_fetched_once_per_conversion():
    """A font referenced by several nodes is fetched only once."""
    tree = parser.Tree(bytestring=svg(rule=True).encode())
    calls = []
    original_fetch = tree.fetch_url

    def counting_fetch(url, resource_type):
        calls.append(url.geturl())
        return original_fetch(url, resource_type)

    tree.fetch_url = counting_fetch
    font_library = FontLibrary(tree)
    assert font_library.get_font_face(
        tree, FONT_ALIAS, cairo.FONT_SLANT_NORMAL,
        cairo.FONT_WEIGHT_NORMAL) is not None
    # A second request reuses the cached result and does not fetch again.
    assert font_library.get_font_face(
        tree, FONT_ALIAS, cairo.FONT_SLANT_NORMAL,
        cairo.FONT_WEIGHT_NORMAL) is not None
    assert len(calls) == 1
    font_library.cleanup()
