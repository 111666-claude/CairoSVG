"""Tests for chunked PNG rendering of very wide/tall SVG documents."""

import io

import pytest
from PIL import Image, ImageChops

from . import svg2png

WIDTH = 33000
HEIGHT = 2

# Two opaque colour bands and a diagonal stroke crossing the canonical
# Cairo surface edge around x == 32767. ``shape-rendering=crispEdges`` keeps
# the band boundaries exact; the diagonal keeps its anti-aliasing.
WIDE_SVG = f"""<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}"
     viewBox="0 0 {WIDTH} {HEIGHT}" preserveAspectRatio="none">
  <rect x="0" y="0" width="{WIDTH}" height="1" fill="#e01e28"/>
  <rect x="0" y="1" width="{WIDTH}" height="1" fill="#1e3ce0"/>
  <line x1="32752" y1="0" x2="32784" y2="{HEIGHT}"
        stroke="#ffffff" stroke-width="1"/>
</svg>
""".encode()


def decode(png_bytes):
    """Decode PNG bytes to a Pillow RGBA image."""
    return Image.open(io.BytesIO(png_bytes)).convert('RGBA')


def render(**kwargs):
    """Render the wide sample and decode it."""
    return decode(svg2png(WIDE_SVG, **kwargs))


def test_automatic_chunking_restores_full_dimensions():
    """An over-edge output is stitched back into one full-size PNG."""
    image = render()
    assert image.size == (WIDTH, HEIGHT)


@pytest.mark.parametrize('chunk_size', [32767, 8192, 4096])
def test_explicit_chunk_sizes_keep_dimensions(chunk_size):
    assert render(chunk_size=chunk_size).size == (WIDTH, HEIGHT)


def test_band_colours_are_exact_at_and_around_the_seam():
    """Opaque bands are placed at output coordinates without offsets."""
    image = render()
    pixels = image.load()
    for x in (100, 5000, 32000, 32766, 32767, 32768, 32900, WIDTH - 1):
        if 32750 <= x <= 32786:
            # Anti-aliased diagonal zone, checked separately.
            continue
        assert pixels[x, 0] == (224, 30, 40, 255)
        assert pixels[x, 1] == (30, 60, 224, 255)


def test_chunk_boundaries_have_no_white_or_transparent_gap():
    """Every seam column stays opaque and band-coloured away from the line."""
    image = render(chunk_size=4096)
    pixels = image.load()
    for seam in range(4096, WIDTH, 4096):
        for x in (seam - 1, seam):
            if 32750 <= x <= 32786:
                continue
            for y in range(HEIGHT):
                red, green, blue, alpha = pixels[x, y]
                assert alpha == 255
                assert not (red > 245 and green > 245 and blue > 245)


def test_diagonal_is_continuous_across_the_cairo_limit():
    """Bright stroke pixels exist on both sides of the 32767 edge."""
    image = render()
    pixels = image.load()

    def bright(xs):
        return any(
            max(pixels[x, y][:3]) > 180
            for x in xs for y in range(HEIGHT))

    assert bright(range(32752, 32767))
    assert bright(range(32767, 32785))


def test_downscaled_chunked_matches_single_render():
    """The stitched output agrees with a small one-surface render."""
    chunked = render()
    small = decode(svg2png(
        WIDE_SVG, output_width=3300, output_height=HEIGHT))
    downscaled = chunked.resize((3300, HEIGHT), Image.BILINEAR)

    total_error = 0
    samples = 0
    for x in range(200, 3200, 5):
        for y in range(HEIGHT):
            first = downscaled.getpixel((x, y))
            second = small.getpixel((x, y))
            total_error += sum(abs(a - b) for a, b in zip(first, second))
            samples += 3
    assert total_error / samples < 12


def test_tall_image_is_chunked_too():
    """Chunking also works when the height exceeds the Cairo edge."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="2" height="33000">'
        '<rect width="2" height="16500" fill="teal"/>'
        '<rect y="16500" width="2" height="16500" fill="maroon"/>'
        '</svg>').encode()
    image = decode(svg2png(svg))
    assert image.size == (2, 33000)
    assert image.getpixel((0, 0)) == (0, 128, 128, 255)
    assert image.getpixel((0, 32999)) == (128, 0, 0, 255)


def test_shapes_transparency_and_gradients_match_single_render():
    """Forced small chunks stay point-for-point close to one surface."""
    svg = b"""<svg xmlns="http://www.w3.org/2000/svg" width="64" height="16">
      <rect x="2" y="2" width="20" height="12" fill="red" opacity="0.5"/>
      <defs>
        <linearGradient id="g" gradientUnits="userSpaceOnUse"
                        x1="0" y1="0" x2="64" y2="0">
          <stop offset="0" stop-color="red"/>
          <stop offset="1" stop-color="blue"/>
        </linearGradient>
      </defs>
      <rect x="28" y="2" width="34" height="12" fill="url(#g)"/>
    </svg>"""
    single = decode(svg2png(svg))
    chunked = decode(svg2png(svg, chunk_size=13))
    assert single.size == chunked.size
    difference = ImageChops.difference(single, chunked)
    # Transparent pixels and gradients are rendered exactly; allow only tiny
    # anti-aliasing coverage differences on shape edges.
    worst = max(high for _, high in difference.getextrema())
    assert worst <= 12


def test_invalid_size_fails_clearly():
    """Zero-size documents keep raising the established ValueError."""
    with pytest.raises(ValueError):
        svg2png(
            b'<svg xmlns="http://www.w3.org/2000/svg" width="0" height="4"/>')


def test_resource_limit_is_enforced():
    """A total-pixel budget is enforced with a clear ValueError."""
    with pytest.raises(ValueError):
        svg2png(WIDE_SVG, max_pixels=1000)


def test_write_to_path_and_buffer(tmp_path):
    """Chunked bytes can be returned or written to path/file-like targets."""
    path = tmp_path / 'wide.png'
    assert svg2png(WIDE_SVG, write_to=str(path)) is None
    assert Image.open(path).size == (WIDTH, HEIGHT)

    buffer = io.BytesIO()
    assert svg2png(WIDE_SVG, write_to=buffer) is None
    assert Image.open(io.BytesIO(buffer.getvalue())).size == (WIDTH, HEIGHT)

    png_bytes = svg2png(WIDE_SVG)
    assert png_bytes[:8] == b'\x89PNG\r\n\x1a\n'
