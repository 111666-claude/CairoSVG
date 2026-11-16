"""
Tests for the ``feComposite`` filter pipeline.

The expected pixels are built with raw Cairo calls rather than hard-coded
constants, so the tests check the SVG semantics (which image is the source
and which one is the backdrop, which region is painted) while still relying
on Cairo's premultiplied-alpha arithmetic.

"""

import io

import cairocffi as cairo
import pytest

from . import svg2png

WIDTH = HEIGHT = 60

# Two translucent shapes used as the source graphic and as the second input:
# a red square and a blue square overlapping on a 20x20 area.
RED = (1, 0, 0, 0.5)
BLUE = (0, 0, 1, 0.6)
RED_RECT = (10, 10, 30, 30)
BLUE_RECT = (20, 20, 30, 30)

OPERATORS = {
    'over': cairo.OPERATOR_OVER,
    'in': cairo.OPERATOR_IN,
    'out': cairo.OPERATOR_OUT,
    'atop': cairo.OPERATOR_ATOP,
    'xor': cairo.OPERATOR_XOR,
}


def render(svg):
    """Render an SVG and return the raw BGRA bytes of the PNG."""
    png = svg2png(bytestring=svg.encode())
    surface = cairo.ImageSurface.create_from_png(io.BytesIO(png))
    return bytes(surface.get_data())


def pixel(data, x, y, width=WIDTH):
    """Return the ``(b, g, r, a)`` byte values of a rendered pixel."""
    index = (y * width + x) * 4
    return tuple(data[index:index + 4])


def composite_reference(operator_name):
    """Build the expected image with raw Cairo compositing calls."""
    surface = cairo.ImageSurface(
        cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    context = cairo.Context(surface)
    # Backdrop (in2): the blue flood square.
    context.set_source_rgba(*BLUE)
    context.rectangle(*BLUE_RECT)
    context.fill()
    # Source (in): the red source graphic composited with the tested
    # operator, matching the SVG operand order.
    context.set_operator(OPERATORS[operator_name])
    context.set_source_rgba(*RED)
    context.rectangle(*RED_RECT)
    context.fill()
    return bytes(surface.get_data())


def composite_svg(operator_name):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}"
        height="{HEIGHT}">
      <defs>
        <filter id="f" x="0" y="0" width="{WIDTH}" height="{HEIGHT}"
            filterUnits="userSpaceOnUse">
          <feFlood flood-color="#0000ff" flood-opacity="0.6"
              x="{BLUE_RECT[0]}" y="{BLUE_RECT[1]}" width="{BLUE_RECT[2]}"
              height="{BLUE_RECT[3]}"/>
          <feComposite in="SourceGraphic" operator="{operator_name}"/>
        </filter>
      </defs>
      <rect x="{RED_RECT[0]}" y="{RED_RECT[1]}" width="{RED_RECT[2]}"
          height="{RED_RECT[3]}" fill="#ff0000" fill-opacity="0.5"
          filter="url(#f)"/>
    </svg>'''


# Points fully covered by the shapes, away from antialiased edges.
SAMPLE_POINTS = (
    (5, 5),      # outside of both shapes
    (15, 15),    # red only
    (25, 25),    # red and blue overlap
    (35, 35),    # red and blue overlap
    (45, 45),    # blue only
    (55, 55),    # outside of both shapes
)


@pytest.mark.parametrize('operator_name', tuple(OPERATORS))
def test_fe_composite_operators(operator_name):
    """The five operators match the SVG premultiplied-alpha compositing."""
    rendered = render(composite_svg(operator_name))
    expected = composite_reference(operator_name)
    for x, y in SAMPLE_POINTS:
        assert pixel(rendered, x, y) == pixel(expected, x, y)


def test_fe_composite_chained_results():
    """Named results can be fed back as in/in2 of later primitives."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80">
      <defs>
        <filter id="f" x="0" y="0" width="80" height="80"
            filterUnits="userSpaceOnUse">
          <feFlood flood-color="#00ff00" flood-opacity="0.5"
              x="10" y="10" width="40" height="40" result="flood"/>
          <feOffset in="flood" dx="20" dy="20" result="shifted"/>
          <feComposite in="SourceGraphic" in2="shifted" operator="in"/>
        </filter>
      </defs>
      <rect x="10" y="10" width="40" height="40" fill="#ff0000"
          fill-opacity="0.8" filter="url(#f)"/>
    </svg>'''
    rendered = render(svg)

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 80, 80)
    context = cairo.Context(surface)
    context.push_group()
    context.set_source_rgba(0, 1, 0, 0.5)
    context.rectangle(10, 10, 40, 40)
    context.fill()
    flood = context.pop_group()
    context.push_group()
    context.translate(20, 20)
    context.set_source(flood)
    context.paint()
    shifted = context.pop_group()
    context.push_group()
    context.set_source_rgba(1, 0, 0, 0.8)
    context.rectangle(10, 10, 40, 40)
    context.fill()
    source = context.pop_group()
    context.set_source(shifted)
    context.paint()
    context.set_operator(cairo.OPERATOR_IN)
    context.set_source(source)
    context.paint()
    expected = bytes(surface.get_data())

    assert pixel(rendered, 40, 40, 80) == pixel(expected, 40, 40, 80)
    assert pixel(rendered, 20, 20, 80) == (0, 0, 0, 0)
    assert pixel(rendered, 60, 60, 80) == (0, 0, 0, 0)


def test_source_alpha():
    """SourceAlpha is a black image with the source alpha channel."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">
      <defs>
        <filter id="f" x="0" y="0" width="40" height="40"
            filterUnits="userSpaceOnUse">
          <feFlood flood-color="#ffffff" flood-opacity="0" result="empty"/>
          <feComposite in="SourceAlpha" in2="empty" operator="over"/>
        </filter>
      </defs>
      <rect x="5" y="5" width="20" height="20" fill="#ff0000"
          fill-opacity="0.5" filter="url(#f)"/>
    </svg>'''
    rendered = render(svg)
    assert pixel(rendered, 15, 15, 40) == (0, 0, 0, 128)
    assert pixel(rendered, 2, 2, 40) == (0, 0, 0, 0)


def test_filter_region_clips_output():
    """Pixels outside of the filter region are left unchanged."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60">
      <rect x="0" y="0" width="60" height="60" fill="#ffffff"/>
      <defs>
        <filter id="f" x="10" y="10" width="20" height="20"
            filterUnits="userSpaceOnUse">
          <feFlood flood-color="#0000ff" flood-opacity="1"/>
          <feComposite in="SourceGraphic" operator="over"/>
        </filter>
      </defs>
      <rect x="0" y="0" width="60" height="60" fill="#ff0000"
          filter="url(#f)"/>
    </svg>'''
    rendered = render(svg)

    # Raw Cairo reference of the filter output: blue flood backdrop, then
    # the opaque red source painted over it, so red wins inside the region.
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 60, 60)
    context = cairo.Context(surface)
    context.set_source_rgba(0, 0, 1, 1)
    context.rectangle(10, 10, 20, 20)
    context.fill()
    context.set_source_rgba(1, 0, 0, 1)
    context.rectangle(10, 10, 20, 20)
    context.fill()
    expected = bytes(surface.get_data())

    assert pixel(rendered, 15, 15) == pixel(expected, 15, 15)
    # Outside of the region: the white background was never overwritten.
    assert pixel(rendered, 5, 5) == (255, 255, 255, 255)
    assert pixel(rendered, 50, 50) == (255, 255, 255, 255)


def test_unfiltered_elements_unchanged():
    """Neighboring opaque and transparent pixels are not modified."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60">
      <rect x="0" y="0" width="30" height="60" fill="#00ff00"/>
      <defs>
        <filter id="f" x="35" y="35" width="25" height="25"
            filterUnits="userSpaceOnUse">
          <feComposite in="SourceGraphic" in2="SourceGraphic"
              operator="over"/>
        </filter>
      </defs>
      <rect x="40" y="40" width="15" height="15" fill="#ff0000"
          filter="url(#f)"/>
    </svg>'''
    rendered = render(svg)
    assert pixel(rendered, 10, 10) == (0, 255, 0, 255)
    assert pixel(rendered, 45, 45) == (0, 0, 255, 255)
    assert pixel(rendered, 35, 35) == (0, 0, 0, 0)


def test_object_bounding_box_units():
    """Plain numbers in objectBoundingBox units are bbox fractions."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80">
      <defs>
        <filter id="f" primitiveUnits="objectBoundingBox"
            x="-20%" y="-20%" width="150%" height="150%">
          <feFlood flood-color="#0000ff" flood-opacity="0.5" result="bg"/>
          <feOffset in="bg" dx="0.25" dy="0.25" result="shifted"/>
          <feComposite in="SourceGraphic" in2="shifted" operator="over"/>
        </filter>
      </defs>
      <rect x="20" y="20" width="40" height="40" fill="#ff0000"
          fill-opacity="0.8" filter="url(#f)"/>
    </svg>'''
    rendered = render(svg)

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 80, 80)
    context = cairo.Context(surface)
    context.set_source_rgba(0, 0, 1, 0.5)
    context.rectangle(30, 30, 40, 40)
    context.fill()
    context.set_operator(cairo.OPERATOR_OVER)
    context.set_source_rgba(1, 0, 0, 0.8)
    context.rectangle(20, 20, 40, 40)
    context.fill()
    expected = bytes(surface.get_data())

    for x, y in ((25, 25), (35, 25), (45, 45)):
        assert pixel(rendered, x, y, 80) == pixel(expected, x, y, 80)


def test_fe_blend_chained_in_pipeline():
    """feBlend can be chained with feComposite inside the pipeline."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60">
      <defs>
        <filter id="f" x="0" y="0" width="60" height="60"
            filterUnits="userSpaceOnUse">
          <feFlood flood-color="#0000ff" flood-opacity="0.6"
              x="20" y="20" width="30" height="30" result="b"/>
          <feBlend in="SourceGraphic" in2="b" mode="multiply"
              result="blended"/>
          <feComposite in="blended" in2="SourceAlpha" operator="in"/>
        </filter>
      </defs>
      <rect x="10" y="10" width="30" height="30" fill="#ff0000"
          fill-opacity="0.5" filter="url(#f)"/>
    </svg>'''
    rendered = render(svg)

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 60, 60)
    context = cairo.Context(surface)
    context.push_group()
    context.set_source_rgba(0, 0, 1, 0.6)
    context.rectangle(20, 20, 30, 30)
    context.fill()
    flood = context.pop_group()
    context.push_group()
    context.set_source(flood)
    context.paint()
    context.set_source_rgba(1, 0, 0, 0.5)
    context.set_operator(cairo.OPERATOR_MULTIPLY)
    context.paint()
    blended = context.pop_group()
    context.push_group()
    context.set_source_rgba(1, 0, 0, 0.5)
    context.rectangle(10, 10, 30, 30)
    context.fill()
    source = context.pop_group()
    context.push_group()
    context.set_source(source)
    context.paint()
    context.set_source_rgba(0, 0, 0, 1)
    context.set_operator(cairo.OPERATOR_IN)
    context.paint()
    source_alpha = context.pop_group()
    context.set_operator(cairo.OPERATOR_OVER)
    context.set_source(source_alpha)
    context.paint()
    context.set_source(blended)
    context.set_operator(cairo.OPERATOR_IN)
    context.paint()
    expected = bytes(surface.get_data())

    for x, y in ((15, 15), (25, 25), (35, 35), (45, 45)):
        assert pixel(rendered, x, y) == pixel(expected, x, y)


def test_missing_reference():
    """A missing input reference raises instead of using a fallback."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">
      <defs>
        <filter id="f" x="0" y="0" width="40" height="40">
          <feComposite in="missing" in2="SourceGraphic" operator="over"/>
        </filter>
      </defs>
      <rect width="10" height="10" filter="url(#f)"/>
    </svg>'''
    with pytest.raises(ValueError, match='not defined'):
        render(svg)


def test_forward_reference():
    """A reference to a later primitive is treated as a missing input."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">
      <defs>
        <filter id="f" x="0" y="0" width="40" height="40">
          <feComposite in="later" in2="SourceGraphic" operator="over"/>
          <feFlood flood-color="red" result="later"/>
        </filter>
      </defs>
      <rect width="10" height="10" filter="url(#f)"/>
    </svg>'''
    with pytest.raises(ValueError, match='not defined'):
        render(svg)


def test_duplicated_result_name():
    """A result defined twice creates a cycle and must be reported."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">
      <defs>
        <filter id="f" x="0" y="0" width="40" height="40">
          <feFlood flood-color="red" result="r"/>
          <feFlood flood-color="blue" result="r"/>
          <feComposite in="SourceGraphic" in2="r" operator="over"/>
        </filter>
      </defs>
      <rect width="10" height="10" filter="url(#f)"/>
    </svg>'''
    with pytest.raises(ValueError, match='duplicated filter result'):
        render(svg)


def test_unsupported_operator():
    """An unsupported operator never degrades to a plain over compositing."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">
      <defs>
        <filter id="f" x="0" y="0" width="40" height="40">
          <feComposite in="SourceGraphic" in2="SourceGraphic"
              operator="arithmetic"/>
        </filter>
      </defs>
      <rect width="10" height="10" filter="url(#f)"/>
    </svg>'''
    with pytest.raises(ValueError, match='unsupported feComposite'):
        render(svg)


def test_unsupported_primitive():
    """An unknown primitive in a pipeline filter is explicitly rejected."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">
      <defs>
        <filter id="f" x="0" y="0" width="40" height="40">
          <feGaussianBlur stdDeviation="2"/>
          <feComposite in="SourceGraphic" in2="SourceGraphic"
              operator="over"/>
        </filter>
      </defs>
      <rect width="10" height="10" filter="url(#f)"/>
    </svg>'''
    with pytest.raises(ValueError, match='unsupported filter primitive'):
        render(svg)
