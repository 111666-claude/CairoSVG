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

    # Read from a pathlib.Path
    assert svg2png(url=temp_0) == expected_content

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


def get_png_pixel(png_bytes, x, y):
    """Return the (r, g, b, a) pixel at (x, y) of a PNG bytestring."""
    image = cairo.ImageSurface.create_from_png(io.BytesIO(png_bytes))
    data = image.get_data()
    offset = y * image.get_stride() + x * 4
    blue, green, red, alpha = data[offset:offset + 4]
    return red, green, blue, alpha


def test_paint_order_parsing():
    """Test paint-order values parsing."""
    from .helpers import parse_paint_order

    normal = ('fill', 'stroke', 'markers')
    assert parse_paint_order('normal') == normal
    assert parse_paint_order('') == normal
    assert parse_paint_order(None) == normal
    # Omitted operations are appended in default order
    assert parse_paint_order('fill') == normal
    assert parse_paint_order('stroke') == ('stroke', 'fill', 'markers')
    assert parse_paint_order('markers') == ('markers', 'fill', 'stroke')
    assert parse_paint_order('stroke fill') == ('stroke', 'fill', 'markers')
    assert parse_paint_order('markers stroke') == (
        'markers', 'stroke', 'fill')
    # Parsing is case-insensitive and ignores extra whitespace
    assert parse_paint_order('  STROKE\tFill ') == (
        'stroke', 'fill', 'markers')
    # Duplicated keywords are invalid: fall back to normal
    assert parse_paint_order('fill fill') == normal
    assert parse_paint_order('stroke fill stroke') == normal
    # Unknown keywords are invalid: fall back to normal
    assert parse_paint_order('unknown') == normal
    assert parse_paint_order('fill unknown') == normal


def test_paint_order_pixels():
    """Test paint-order rendering with a semi-transparent thick stroke."""
    svg = b'''<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
      <path d="M 20 20 L 80 20 L 80 80 L 20 80 Z" fill="rgb(255,0,0)"
            stroke="rgba(0,0,255,0.5)" stroke-width="20" %s/>
    </svg>'''

    # Normal order: the stroke is painted over the fill
    normal = svg2png(bytestring=svg % b'')
    red, green, blue, alpha = get_png_pixel(normal, 25, 50)
    assert red < 255 and blue > 0  # fill darkened by the stroke on top

    # Stroke before fill: the fill covers the inner half of the stroke
    stroke_fill = svg2png(bytestring=svg % b'paint-order="stroke fill"')
    assert get_png_pixel(stroke_fill, 25, 50) == (255, 0, 0, 255)

    # The outer half of the stroke is identical in both orders
    assert get_png_pixel(stroke_fill, 15, 50) == get_png_pixel(normal, 15, 50)
    # The center of the path is identical in both orders
    assert get_png_pixel(stroke_fill, 50, 50) == (255, 0, 0, 255)
    assert get_png_pixel(normal, 50, 50) == (255, 0, 0, 255)
    # Outside of the stroke, nothing is painted
    assert get_png_pixel(stroke_fill, 5, 50) == (0, 0, 0, 0)

    # "stroke" alone means "stroke fill markers"
    stroke_only = svg2png(bytestring=svg % b'paint-order="stroke"')
    assert get_png_pixel(stroke_only, 25, 50) == (255, 0, 0, 255)

    # Invalid values fall back to the normal order
    invalid = svg2png(bytestring=svg % b'paint-order="stroke stroke"')
    assert get_png_pixel(invalid, 25, 50) == get_png_pixel(normal, 25, 50)
    unknown = svg2png(bytestring=svg % b'paint-order="fill bogus"')
    assert get_png_pixel(unknown, 25, 50) == get_png_pixel(normal, 25, 50)


def test_paint_order_inheritance():
    """Test that paint-order is inherited from surrounding groups."""
    svg = b'''<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
      <g paint-order="stroke fill">
        <path d="M 20 20 L 80 20 L 80 80 L 20 80 Z" fill="rgb(255,0,0)"
              stroke="rgba(0,0,255,0.5)" stroke-width="20"/>
      </g>
    </svg>'''
    assert get_png_pixel(svg2png(bytestring=svg), 25, 50) == (255, 0, 0, 255)


def test_paint_order_opacity():
    """Test that opacity applies to the whole reordered painting."""
    svg = b'''<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
      <path d="M 20 20 L 80 20 L 80 80 L 20 80 Z" fill="rgb(255,0,0)"
            stroke="rgb(0,0,255)" stroke-width="20" opacity="0.5"
            paint-order="stroke fill"/>
    </svg>'''
    red, green, blue, alpha = get_png_pixel(svg2png(bytestring=svg), 25, 50)
    # Leaf opacity is folded into fill and stroke opacity: the 50% red fill
    # is painted over the 50% blue stroke (pixels are premultiplied)
    assert 120 < red < 135 and 55 < blue < 70
    assert 185 < alpha < 200  # both semi-transparent paints combined


def test_paint_order_markers():
    """Test that markers can be painted below fill and stroke."""
    svg = b'''<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
      <defs>
        <marker id="mark" markerUnits="userSpaceOnUse"
                refX="0" refY="0" markerWidth="30" markerHeight="30">
          <circle cx="0" cy="0" r="15" fill="rgb(255,0,0)"/>
        </marker>
      </defs>
      <path d="M 20 50 L 50 50 L 80 50" fill="none" stroke="rgb(0,0,255)"
            stroke-width="20" marker-mid="url(#mark)" %s/>
    </svg>'''

    # Normal order: markers are painted over the stroke
    normal = svg2png(bytestring=svg % b'')
    assert get_png_pixel(normal, 50, 50) == (255, 0, 0, 255)

    # Markers first: the stroke covers the marker
    markers_first = svg2png(bytestring=svg % b'paint-order="markers"')
    assert get_png_pixel(markers_first, 50, 50) == (0, 0, 255, 255)

    # Elements without markers are not affected by the "markers" keyword
    no_marker = b'''<svg xmlns="http://www.w3.org/2000/svg"
         width="100" height="100">
      <path d="M 20 20 L 80 20 L 80 80 L 20 80 Z" fill="rgb(255,0,0)"
            stroke="rgba(0,0,255,0.5)" stroke-width="20"
            paint-order="markers stroke"/>
    </svg>'''
    assert get_png_pixel(svg2png(bytestring=no_marker), 25, 50) == (
        255, 0, 0, 255)
