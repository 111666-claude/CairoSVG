"""
CairoSVG API test suite.

These tests can be used as deployment tests.

"""

import io
import sys

import cairocffi as cairo
import pytest

from . import SURFACES, VERSION, parser, surface, svg2pdf, svg2png
from .__main__ import main
from .url import URLFetchingError

MAGIC_NUMBERS = {
    'SVG': b'<?xml',
    'PNG': b'\211PNG\r\n\032\n',
    'PDF': b'%PDF',
    'PS': b'%!',
}

SVG_SAMPLE = b'''\
<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="4in" height="5in">
  <rect x="5" y="10" width="13" height="15"
        fill="lime" stroke="black" stroke-width="1" />
</svg>
'''


@pytest.mark.parametrize('format_name', MAGIC_NUMBERS)
def test_formats(format_name):
    """Convert to a given format and test that output looks right."""
    content = SURFACES[format_name].convert(SVG_SAMPLE)
    assert content.startswith(MAGIC_NUMBERS[format_name])


def read_file(filename):
    """Shortcut to return the whole content of a file as a byte string."""
    with open(filename, 'rb') as file_object:
        return file_object.read()


def test_api(tmp_path):
    """Test the Python API with various parameters."""
    expected_content = svg2png(SVG_SAMPLE)
    # Already tested above: just a sanity check:
    assert expected_content.startswith(MAGIC_NUMBERS['PNG'])

    # Read from a byte string
    assert svg2png(SVG_SAMPLE) == expected_content
    assert svg2png(bytestring=SVG_SAMPLE) == expected_content

    file_like = io.BytesIO(SVG_SAMPLE)
    # Read from a file-like object
    assert svg2png(file_obj=file_like) == expected_content

    file_like = io.BytesIO()
    # Write to a file-like object
    svg2png(SVG_SAMPLE, write_to=file_like)
    assert file_like.getvalue() == expected_content

    temp_0 = tmp_path / 'sample_0.svg'
    temp_0.write_bytes(SVG_SAMPLE)

    # Read from a filename
    assert svg2png(url=str(temp_0)) == expected_content
    assert svg2png(url=f'file://{temp_0}') == expected_content

    with temp_0.open('rb') as file_object:
        # Read from a real file object
        assert svg2png(file_obj=file_object) == expected_content

    temp_1 = tmp_path / 'result_1.png'
    with temp_1.open('wb') as file_object:
        # Write to a real file object
        svg2png(SVG_SAMPLE, write_to=file_object)
    assert read_file(str(temp_1)) == expected_content

    temp_2 = tmp_path / 'result_2.png'
    # Write to a filename
    svg2png(SVG_SAMPLE, write_to=str(temp_2))
    assert read_file(str(temp_2)) == expected_content

    file_like = io.BytesIO()
    try:
        svg2png(write_to=file_like)
    except TypeError:
        pass
    else:  # pragma: no cover
        raise Exception('TypeError not raised')


def test_low_level_api():
    """Test the low-level Python API with various parameters."""
    expected_content = svg2png(SVG_SAMPLE)

    # Same as above, longer version
    tree = parser.Tree(bytestring=SVG_SAMPLE)
    file_like = io.BytesIO()
    png_surface = surface.PNGSurface(tree, file_like, 96)
    png_surface.finish()
    assert file_like.getvalue() == expected_content

    png_result = cairo.ImageSurface.create_from_png(
        io.BytesIO(expected_content))
    expected_width = png_result.get_width()
    expected_height = png_result.get_height()

    # Abstract surface
    png_surface = surface.PNGSurface(tree, None, 96)
    assert png_surface.width == expected_width
    assert png_surface.height == expected_height
    assert cairo.SurfacePattern(png_surface.cairo)


def test_script(tmp_path):
    """Test the ``cairosvg`` script and the ``main`` function."""
    expected_png = svg2png(SVG_SAMPLE)[:100]
    expected_pdf = svg2pdf(SVG_SAMPLE)[:100]

    def test_main(args, exit_=False, input_=None, full=False):
        """Test main called with given ``args``.

        If ``exit_`` is ``True``, check that ``SystemExit`` is raised. We then
        assume that the program output is an unicode string.

        If ``input_`` is given, use this stream as input stream.

        """
        sys.argv = ['cairosvg'] + args
        old_stdin, old_stdout = sys.stdin, sys.stdout

        output_buffer = io.BytesIO()
        sys.stdout = io.TextIOWrapper(output_buffer)

        if input_:
            sys.stdin = open(input_, 'rb')
            sys.stdin.buffer = sys.stdin

        if exit_:
            try:
                main()
            except SystemExit:
                pass
            else:  # pragma: no cover
                raise Exception('CairoSVG did not exit')
        else:
            main()

        sys.stdout.flush()
        output = output_buffer.getvalue()
        sys.stdin, sys.stdout = old_stdin, old_stdout
        return output if full else output[:100]

    svg_path = tmp_path / "test.svg"
    svg_path.write_bytes(SVG_SAMPLE)
    svg_filename = str(svg_path)

    assert test_main(['--help'], exit_=True).startswith(b'usage: ')
    assert test_main(['--version'], exit_=True).strip() == (
        VERSION.encode('ascii'))

    assert test_main([svg_filename]) == expected_pdf
    assert test_main([svg_filename, '-d', '96', '-f', 'pdf']) == expected_pdf
    assert test_main([svg_filename, '-f', 'png']) == expected_png
    assert test_main(['-'], input_=svg_filename) == expected_pdf

    # Test DPI
    output = test_main([svg_filename, '-d', '10', '-f', 'png'], full=True)
    image = cairo.ImageSurface.create_from_png(io.BytesIO(output))
    assert image.get_width() == 40
    assert image.get_height() == 50

    temp_1 = tmp_path / 'result_1'
    # Default to PDF
    assert not test_main([svg_filename, '-o', str(temp_1)])
    assert read_file(temp_1)[:100] == expected_pdf

    temp_2 = tmp_path / 'result_2.png'
    # Guess from the file extension
    assert not test_main([svg_filename, '-o', str(temp_2)])
    assert read_file(temp_2)[:100] == expected_png

    temp_3 = tmp_path / 'result_3.png'
    # Explicit -f wins
    assert not test_main([svg_filename, '-o', str(temp_3), '-f', 'pdf'])
    assert read_file(str(temp_3))[:100] == expected_pdf


# A simple SVG whose only shape would be black without an external stylesheet
EXTERNAL_CSS_SVG = b'''\
<?xml version="1.0"?>
<?xml-stylesheet type="text/css" href="style.css"?>
<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
  <rect id="square" width="10" height="10"/>
</svg>
'''

EXTERNAL_CSS = b'rect { fill: #ff0000; }\n'


def _center_pixel(png_bytes):
    """Return the opaque ``(r, g, b, a)`` color of the center PNG pixel."""
    image = cairo.ImageSurface.create_from_png(io.BytesIO(png_bytes))
    data = image.get_data()
    width = image.get_width()
    height = image.get_height()
    # ARGB32 is stored as BGRA on little-endian machines
    offset = (height // 2) * image.get_stride() + (width // 2) * 4
    b, g, r, a = data[offset:offset + 4]
    return r, g, b, a


def test_external_stylesheet_local_file(tmp_path):
    """An xml-stylesheet PI applies CSS stored in a sibling file."""
    svg_path = tmp_path / 'sample.svg'
    svg_path.write_bytes(EXTERNAL_CSS_SVG)
    (tmp_path / 'style.css').write_bytes(EXTERNAL_CSS)

    # Without the external stylesheet the rect is black
    black_svg = EXTERNAL_CSS_SVG.split(b'\n', 2)[2]
    assert _center_pixel(svg2png(black_svg))[:3] == (0, 0, 0)

    # With unsafe=True the local stylesheet is found and applied
    png = svg2png(url=str(svg_path), unsafe=True)
    assert _center_pixel(png)[:3] == (255, 0, 0)


def test_external_stylesheet_relative_href(tmp_path):
    """Relative hrefs are resolved from the SVG file location."""
    assets = tmp_path / 'assets'
    assets.mkdir()
    (assets / 'style.css').write_bytes(b'rect { fill: rgb(0, 128, 0); }')
    svg_path = tmp_path / 'nested' / 'sample.svg'
    svg_path.parent.mkdir()
    svg_path.write_bytes(
        EXTERNAL_CSS_SVG.replace(b'href="style.css"',
                                 b'href="../assets/style.css"'))

    png = svg2png(url=str(svg_path), unsafe=True)
    assert _center_pixel(png)[:3] == (0, 128, 0)


def test_external_stylesheet_safe_mode_blocks_local_file(tmp_path):
    """Default safe mode never silently opens a local external stylesheet."""
    svg_path = tmp_path / 'sample.svg'
    svg_path.write_bytes(EXTERNAL_CSS_SVG)
    (tmp_path / 'style.css').write_bytes(EXTERNAL_CSS)

    with pytest.raises(URLFetchingError, match='safe mode'):
        svg2png(url=str(svg_path))


def test_external_stylesheet_safe_mode_blocks_network():
    """Default safe mode does not fetch stylesheets over the network."""
    svg = EXTERNAL_CSS_SVG.replace(
        b'href="style.css"',
        b'href="http://example.invalid/style.css"')
    with pytest.raises(URLFetchingError, match='safe mode'):
        svg2png(svg)


def test_external_stylesheet_missing_resource(tmp_path):
    """A referenced stylesheet that does not exist raises an explicit error."""
    svg_path = tmp_path / 'sample.svg'
    svg_path.write_bytes(EXTERNAL_CSS_SVG)

    with pytest.raises(URLFetchingError, match='Could not fetch'):
        svg2png(url=str(svg_path), unsafe=True)


def test_external_stylesheet_data_url_safe_mode():
    """data: URL stylesheets are allowed by the default safe fetcher."""
    svg = b'''\
<?xml version="1.0"?>
<?xml-stylesheet type="text/css"
    href="data:text/css;charset=utf-8,rect%20%7B%20fill%3A%20blue%3B%20%7D"?>
<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
  <rect width="10" height="10"/>
</svg>
'''
    assert _center_pixel(svg2png(svg))[:3] == (0, 0, 255)


def _render_png(svg_path, **kwargs):
    """Render an SVG file to PNG bytes through the low-level surface API."""
    tree = parser.Tree(url=str(svg_path), **kwargs)
    file_obj = io.BytesIO()
    png_surface = surface.PNGSurface(tree, file_obj, 96)
    png_surface.finish()
    return file_obj.getvalue()


def test_external_stylesheet_custom_url_fetcher(tmp_path):
    """A user-provided URL fetcher is used for external stylesheets."""
    svg_path = tmp_path / 'sample.svg'
    svg_path.write_bytes(EXTERNAL_CSS_SVG)
    (tmp_path / 'style.css').write_bytes(b'rect { fill: purple; }')

    def fetcher(url, resource_type):
        with open(url.removeprefix('file://'), 'rb') as file_obj:
            return file_obj.read()

    png = _render_png(svg_path, url_fetcher=fetcher)
    assert _center_pixel(png)[:3] == (128, 0, 128)


def test_external_stylesheet_cascade_order(tmp_path):
    """External CSS, inline style and presentation attributes cascade."""
    svg_path = tmp_path / 'sample.svg'
    (tmp_path / 'style.css').write_bytes(
        b'rect { fill: red; stroke: blue; }\n')
    # Presentation attribute (lowest priority) loses to the external CSS,
    # the inline style attribute (highest priority) wins over the CSS.
    svg_path.write_text('''\
<?xml version="1.0"?>
<?xml-stylesheet type="text/css" href="style.css"?>
<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
  <rect width="10" height="10" fill="lime"
        style="fill: #00ffff"/>
</svg>
''')
    assert _center_pixel(svg2png(url=str(svg_path), unsafe=True))[:3] == (
        0, 255, 255)

    # Without the inline style attribute, the external CSS beats the
    # presentation attribute.
    svg_path.write_text('''\
<?xml version="1.0"?>
<?xml-stylesheet type="text/css" href="style.css"?>
<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
  <rect width="10" height="10" fill="lime"/>
</svg>
''')
    assert _center_pixel(svg2png(url=str(svg_path), unsafe=True))[:3] == (
        255, 0, 0)


def test_external_stylesheet_duplicate_pi(tmp_path):
    """Two PIs referencing the same stylesheet fetch it only once."""
    svg_path = tmp_path / 'sample.svg'
    svg_path.write_bytes(b'''\
<?xml version="1.0"?>
<?xml-stylesheet type="text/css" href="style.css"?>
<?xml-stylesheet type="text/css" href="style.css"?>
<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
  <rect width="10" height="10"/>
</svg>
''')
    calls = []

    def fetcher(url, resource_type):
        calls.append((url, resource_type))
        if resource_type == 'text/css':
            return b'rect { fill: teal; }'
        with open(url.removeprefix('file://'), 'rb') as file_obj:
            return file_obj.read()

    png = _render_png(svg_path, url_fetcher=fetcher)
    assert len([call for call in calls if call[1] == 'text/css']) == 1
    assert _center_pixel(png)[:3] == (0, 128, 128)


def test_no_stylesheet_pi_unchanged():
    """Documents without an xml-stylesheet PI render exactly as before."""
    svg = b'''\
<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
  <rect width="10" height="10" fill="orange"/>
</svg>
'''
    assert _center_pixel(svg2png(svg))[:3] == (255, 165, 0)
    assert svg2png(svg) == svg2png(svg)


def test_alternate_stylesheets_ignored(tmp_path):
    """Alternate (titled) stylesheets are not applied; only the preferred."""
    svg_path = tmp_path / 'sample.svg'
    svg_path.write_bytes(b'''\
<?xml version="1.0"?>
<?xml-stylesheet type="text/css" href="preferred.css" title="A"?>
<?xml-stylesheet alternate="yes" type="text/css"
    href="alternate.css" title="B"?>
<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
  <rect width="10" height="10"/>
</svg>
''')
    (tmp_path / 'preferred.css').write_bytes(b'rect { fill: red; }')
    (tmp_path / 'alternate.css').write_bytes(b'rect { fill: green; }')

    def fetcher(url, resource_type):
        with open(url.removeprefix('file://'), 'rb') as file_obj:
            return file_obj.read()

    assert _center_pixel(
        _render_png(svg_path, url_fetcher=fetcher))[:3] == (255, 0, 0)
