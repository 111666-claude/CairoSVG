"""
CairoSVG API test suite.

These tests can be used as deployment tests.

"""

import io
import struct
import sys

import cairocffi as cairo
import pytest

from . import (
    SURFACES, VERSION, parser, surface, svg2pdf, svg2png, svg2surface)
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


# Fixed opaque shapes: white background, red circle, lime rectangle. Every
# pixel is fully opaque, so expected channel values are simple and exact.
OPAQUE_SVG = b'''\
<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"
     viewBox="0 0 8 8">
  <rect width="8" height="8" fill="white"/>
  <rect x="0" y="0" width="4" height="4" fill="lime"/>
  <circle cx="6" cy="6" r="2" fill="red"/>
</svg>
'''

# Fixed translucent shapes over a transparent background, exercising
# premultiplied alpha and fully transparent pixels.
TRANSLUCENT_SVG = b'''\
<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"
     viewBox="0 0 8 8">
  <rect x="0" y="0" width="4" height="4" fill="red" opacity="0.5"/>
  <rect x="4" y="4" width="4" height="4" fill="lime" fill-opacity="0.25"/>
</svg>
'''


def png_surface(bytestring, **kwargs):
    """Render an SVG with ``svg2png`` and decode the PNG to a surface."""
    return cairo.ImageSurface.create_from_png(
        io.BytesIO(svg2png(bytestring, **kwargs)))


def pixel_bytes(image_surface):
    """Flush the surface and return its raw pixel buffer as bytes."""
    image_surface.flush()
    return bytes(image_surface.get_data())


def argb_pixel(image_surface, x, y):
    """Return the native-endian ``0xAARRGGBB`` word of one pixel."""
    data = pixel_bytes(image_surface)
    offset = image_surface.get_stride() * y + 4 * x
    return struct.unpack_from('=I', data, offset)[0]


def test_surface_matches_png_opaque():
    """Opaque fixed shapes: buffer equals the decoded PNG pixel by pixel."""
    result = svg2surface(OPAQUE_SVG)
    reference = png_surface(OPAQUE_SVG)

    assert isinstance(result, cairo.ImageSurface)
    assert result.get_format() == cairo.FORMAT_ARGB32
    assert (result.get_width(), result.get_height()) == (8, 8)
    assert result.get_stride() == (
        cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, 8))

    # Exact, point-for-point comparison against the decoded PNG output.
    assert pixel_bytes(result) == pixel_bytes(reference)

    # Every pixel is fully opaque.
    for y in range(8):
        for x in range(8):
            assert argb_pixel(result, x, y) & 0xFF000000 == 0xFF000000

    # Spot-check fixed shapes (0xAARRGGBB words).
    assert argb_pixel(result, 1, 1) == 0xFF00FF00  # lime
    assert argb_pixel(result, 6, 6) == 0xFFFF0000  # red
    assert argb_pixel(result, 4, 0) == 0xFFFFFFFF  # white background


def test_surface_matches_png_translucent():
    """Translucent fixed shapes: premultiplied buffer matches decoded PNG."""
    result = svg2surface(TRANSLUCENT_SVG)
    reference = png_surface(TRANSLUCENT_SVG)

    # Exact, point-for-point comparison, including anti-aliased edges and
    # fully transparent pixels.
    assert pixel_bytes(result) == pixel_bytes(reference)

    # Red with opacity 0.5: premultiplied R = 128, A = 128.
    assert argb_pixel(result, 1, 1) == 0x80800000
    # Lime with fill-opacity 0.25: premultiplied G = 64, A = 64.
    assert argb_pixel(result, 5, 5) == 0x40004000
    # Untouched corners stay fully transparent.
    assert argb_pixel(result, 0, 7) == 0x00000000
    assert argb_pixel(result, 7, 0) == 0x00000000

    # On little-endian systems the byte order is premultiplied BGRA.
    if sys.byteorder == 'little':
        data = pixel_bytes(result)
        assert data[0:4] == bytes((0, 0, 128, 128))


def test_surface_background_color_and_scale():
    """Background color and scaling behave exactly like ``svg2png``."""
    by_kwargs = dict(background_color='blue', scale=2)
    result = svg2surface(TRANSLUCENT_SVG, **by_kwargs)
    reference = png_surface(TRANSLUCENT_SVG, **by_kwargs)

    assert (result.get_width(), result.get_height()) == (16, 16)
    assert pixel_bytes(result) == pixel_bytes(reference)
    # The blue background shows through the transparent corner.
    assert argb_pixel(result, 15, 0) == 0xFF0000FF

    # Explicit output dimensions follow the same rules as svg2png.
    sized = svg2surface(OPAQUE_SVG, output_width=4, output_height=4)
    assert (sized.get_width(), sized.get_height()) == (4, 4)
    assert pixel_bytes(sized) == pixel_bytes(
        png_surface(OPAQUE_SVG, output_width=4, output_height=4))


def test_surface_repeated_calls_are_independent():
    """Consecutive calls return distinct, isolated surfaces."""
    first = svg2surface(OPAQUE_SVG)
    first_data = pixel_bytes(first)
    second = svg2surface(TRANSLUCENT_SVG)
    third = svg2surface(OPAQUE_SVG)

    # Distinct underlying objects and independent buffer contents.
    assert first is not second is not third
    assert pixel_bytes(third) == first_data
    assert pixel_bytes(second) != first_data

    # Mutating one surface does not affect surfaces from other calls: the
    # buffers are independent allocations.
    buffer = second.get_data()
    buffer[0:1] = b'\xff'
    second.mark_dirty()
    assert pixel_bytes(third) == first_data
    assert pixel_bytes(first) == first_data

    # Finishing and releasing a surface does not break the next conversion.
    second.finish()
    del second
    again = svg2surface(TRANSLUCENT_SVG)
    assert argb_pixel(again, 1, 1) == 0x80800000
    assert pixel_bytes(first) == first_data


def test_surface_low_level_api():
    """The class method behind ``svg2surface`` renders without a PNG target."""
    image_surface = surface.PNGSurface.convert_to_surface(
        bytestring=OPAQUE_SVG)
    assert isinstance(image_surface, cairo.ImageSurface)
    assert pixel_bytes(image_surface) == pixel_bytes(png_surface(OPAQUE_SVG))


@pytest.mark.parametrize('svg, exception', [
    (b'<svg xmlns="http://www.w3.org/2000/svg"></svg>', ValueError),
    (b'this is not an SVG document', Exception),
])
def test_surface_invalid_input(svg, exception):
    """Invalid inputs raise the same exceptions as the PNG converter."""
    with pytest.raises(exception):
        svg2png(svg)
    with pytest.raises(exception):
        svg2surface(svg)
