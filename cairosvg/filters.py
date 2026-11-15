"""
SVG filter primitives.

This module implements the raster part of the SVG filter model. Filters that
can only be expressed by transforming pixels (starting with
``feColorMatrix``) are evaluated on Cairo image surfaces and combined with the
already supported vector primitives (``feOffset``, ``feBlend`` and
``feFlood``).

"""

import math
import re
import struct

from .bounding_box import calculate_bounding_box, is_non_empty_bounding_box
from .helpers import normalize, paint, size
from .surface import cairo

# Cairo operators matching the SVG feBlend mode names
BLEND_OPERATORS = {
    'darken': cairo.OPERATOR_DARKEN,
    'lighten': cairo.OPERATOR_LIGHTEN,
    'multiply': cairo.OPERATOR_MULTIPLY,
    'normal': cairo.OPERATOR_OVER,
    'screen': cairo.OPERATOR_SCREEN,
}

# Rec. 709 luminance coefficients, as mandated by the SVG specification
LUMINANCE_RED = 0.2126
LUMINANCE_GREEN = 0.7152
LUMINANCE_BLUE = 0.0722

# Identity color matrix, used as the default value of the "matrix" type
IDENTITY_MATRIX = (
    1, 0, 0, 0, 0,
    0, 1, 0, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 0, 1, 0,
)

# luminanceToAlpha: alpha becomes the Rec. 709 luminance of the source color
LUMINANCE_TO_ALPHA_MATRIX = (
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    LUMINANCE_RED, LUMINANCE_GREEN, LUMINANCE_BLUE, 0, 0,
)


def filter_uses_color_matrix(surface, name):
    """Check whether ``name`` contains a pixel-based filter primitive."""
    filter_node = surface.filters.get(name)
    if filter_node is None:
        return False
    return any(child.tag == 'feColorMatrix' for child in filter_node.children)


def _tokenize(string):
    """Split an SVG numeric attribute into float tokens."""
    if not string:
        return []
    return [float(token) for token in normalize(string).split()]


_ANGLE_RE = re.compile(
    r'[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?(?:turn|deg|grad|rad)?',
    re.IGNORECASE)


def _parse_angle(value):
    """Parse an angle expressed in deg, grad, rad or turn (default: deg)."""
    if value is None:
        return 0
    if not isinstance(value, str):
        # Bare numeric values are expressed in degrees.
        return math.radians(value)
    string = value.strip()
    for unit, factor in (
            ('turn', 360), ('deg', 1), ('grad', 0.9), ('rad', 180 / math.pi)):
        if string.endswith(unit):
            return math.radians(float(string[:-len(unit)]) * factor)
    return math.radians(float(string) if string else 0)


def _parse_color_matrix(child):
    """Return the 4x5 RGBA matrix of an ``feColorMatrix`` node.

    Raise :class:`ValueError` for an unknown type, a wrong number of values or
    for coefficients that are not finite numbers.

    """
    matrix_type = child.get('type', 'matrix')

    if matrix_type == 'matrix':
        values = _tokenize(child.get('values'))
        if not values:
            return IDENTITY_MATRIX
        if len(values) != 20:
            raise ValueError(
                'feColorMatrix of type "matrix" requires 20 values, got '
                f'{len(values)}')
        matrix = values

    elif matrix_type == 'saturate':
        values = _tokenize(child.get('values'))
        if not values:
            saturation = 1
        elif len(values) != 1:
            raise ValueError(
                'feColorMatrix of type "saturate" requires 1 value, got '
                f'{len(values)}')
        else:
            saturation = values[0]
        matrix = (
            LUMINANCE_RED * (1 - saturation) + saturation,
            LUMINANCE_GREEN * (1 - saturation),
            LUMINANCE_BLUE * (1 - saturation), 0, 0,
            LUMINANCE_RED * (1 - saturation),
            LUMINANCE_GREEN * (1 - saturation) + saturation,
            LUMINANCE_BLUE * (1 - saturation), 0, 0,
            LUMINANCE_RED * (1 - saturation),
            LUMINANCE_GREEN * (1 - saturation),
            LUMINANCE_BLUE * (1 - saturation) + saturation, 0, 0,
            0, 0, 0, 1, 0,
        )

    elif matrix_type == 'hueRotate':
        angle_tokens = _ANGLE_RE.findall(child.get('values') or '')
        if not angle_tokens:
            angle = 0
        elif len(angle_tokens) != 1:
            raise ValueError(
                'feColorMatrix of type "hueRotate" requires 1 value, got '
                f'{len(angle_tokens)}')
        else:
            angle = _parse_angle(angle_tokens[0])
        cosine = math.cos(angle)
        sine = math.sin(angle)
        matrix = (
            LUMINANCE_RED + cosine * (1 - LUMINANCE_RED)
            + sine * (-LUMINANCE_RED),
            LUMINANCE_GREEN + cosine * (-LUMINANCE_GREEN)
            + sine * (-LUMINANCE_GREEN),
            LUMINANCE_BLUE + cosine * (-LUMINANCE_BLUE)
            + sine * (1 - LUMINANCE_BLUE),
            0, 0,
            LUMINANCE_RED + cosine * (-LUMINANCE_RED) + sine * 0.143,
            LUMINANCE_GREEN + cosine * (1 - LUMINANCE_GREEN)
            + sine * 0.140,
            LUMINANCE_BLUE + cosine * (-LUMINANCE_BLUE) + sine * -0.283,
            0, 0,
            LUMINANCE_RED + cosine * (-LUMINANCE_RED)
            + sine * (-(1 - LUMINANCE_RED)),
            LUMINANCE_GREEN + cosine * (-LUMINANCE_GREEN)
            + sine * LUMINANCE_GREEN,
            LUMINANCE_BLUE + cosine * (1 - LUMINANCE_BLUE)
            + sine * LUMINANCE_BLUE,
            0, 0,
            0, 0, 0, 1, 0,
        )

    elif matrix_type == 'luminanceToAlpha':
        matrix = LUMINANCE_TO_ALPHA_MATRIX

    else:
        raise ValueError(f'Unknown feColorMatrix type "{matrix_type}"')

    if any(not math.isfinite(value) for value in matrix):
        raise ValueError('feColorMatrix coefficients must be finite numbers')
    return matrix


def _new_image_surface(width, height):
    """Create a transparent ARGB image surface."""
    return cairo.ImageSurface(
        cairo.FORMAT_ARGB32, int(round(width)), int(round(height)))


def _rasterize_pattern(pattern, width, height):
    """Paint a Cairo pattern on a new image surface using identity matrix."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    new_context = cairo.Context(surface)
    new_context.set_source(pattern)
    new_context.paint()
    surface.flush()
    return surface


def _device_rectangle(context, x, y, width, height):
    """Transform a user-space rectangle into device pixel bounds."""
    left, top = context.user_to_device(x, y)
    right, bottom = context.user_to_device(x + width, y + height)
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top
    return left, top, right, bottom


def _filter_length(surface, string, reference):
    """Parse a filter coordinate in user space.

    Unlike :func:`cairosvg.helpers.size`, unitless values are treated as user
    units (the required behavior for ``userSpaceOnUse`` filter coordinates)
    and percentages as fractions of ``reference``.

    """
    if not string:
        return 0
    string = string.strip()
    if string.endswith('%'):
        return float(string[:-1]) * reference / 100
    if string.endswith('em'):
        return surface.font_size * float(string[:-2])
    if string.endswith('ex') or string.endswith('ch'):
        return surface.font_size * float(string[:-2]) / 2
    for unit, coefficient in (
            ('mm', 1 / 25.4), ('cm', 1 / 2.54), ('in', 1),
            ('pt', 1 / 72), ('pc', 1 / 6)):
        if string.endswith(unit):
            return float(string[:-len(unit)]) * surface.dpi * coefficient
    try:
        return float(string)
    except ValueError:
        return 0


def _pixel_bounds(context, left, top, right, bottom, width, height):
    """Clamp a device-space rectangle to the raster surface, rounding out."""
    left, top, right, bottom = _device_rectangle(
        context, left, top, right - left, bottom - top)
    return (
        max(0, int(math.floor(left))),
        max(0, int(math.floor(top))),
        min(width, int(math.ceil(right))),
        min(height, int(math.ceil(bottom))))


def _subregion_user_space(surface, node, filter_node, child):
    """Return the primitive subregion as a user-space ``(x, y, w, h)``."""
    if filter_node.get('primitiveUnits') == 'objectBoundingBox':
        bounding_box = calculate_bounding_box(surface, node)
        if bounding_box and is_non_empty_bounding_box(bounding_box):
            box_x, box_y, box_width, box_height = bounding_box
        else:
            box_x = box_y = box_width = box_height = 0
        x = box_x + size(surface, child.get('x', '0'), 1) * box_width
        y = box_y + size(surface, child.get('y', '0'), 1) * box_height
        sub_width = size(surface, child.get('width', '100%'), 1) * box_width
        sub_height = (
            size(surface, child.get('height', '100%'), 1) * box_height)
    else:
        x = _filter_length(
            surface, child.get('x', '-10%'), surface.context_width)
        y = _filter_length(
            surface, child.get('y', '-10%'), surface.context_height)
        sub_width = _filter_length(
            surface, child.get('width', '120%'), surface.context_width)
        sub_height = _filter_length(
            surface, child.get('height', '120%'), surface.context_height)
    return x, y, sub_width, sub_height


def _primitive_subregion(surface, node, filter_node, child, width, height,
                         context):
    """Return the device pixel bounds of a filter primitive subregion."""
    x, y, sub_width, sub_height = _subregion_user_space(
        surface, node, filter_node, child)
    return _pixel_bounds(
        context, x, y, x + sub_width, y + sub_height, width, height)


def _filter_region(surface, node, filter_node, width, height, context):
    """Return the device pixel bounds of the whole filter region."""
    # filterUnits defaults to objectBoundingBox per the SVG specification.
    if filter_node.get('filterUnits', 'objectBoundingBox') == \
            'objectBoundingBox':
        bounding_box = calculate_bounding_box(surface, node)
        if bounding_box and is_non_empty_bounding_box(bounding_box):
            box_x, box_y, box_width, box_height = bounding_box
        else:
            box_x = box_y = box_width = box_height = 0
        x = box_x + size(surface, filter_node.get('x', '-10%'), 1) * box_width
        y = box_y + size(surface, filter_node.get('y', '-10%'), 1) * box_height
        region_width = (
            size(surface, filter_node.get('width', '120%'), 1) * box_width)
        region_height = (
            size(surface, filter_node.get('height', '120%'), 1) * box_height)
    else:
        x = _filter_length(
            surface, filter_node.get('x', '-10%'), surface.context_width)
        y = _filter_length(
            surface, filter_node.get('y', '-10%'), surface.context_height)
        region_width = _filter_length(
            surface, filter_node.get('width', '120%'),
            surface.context_width)
        region_height = _filter_length(
            surface, filter_node.get('height', '120%'),
            surface.context_height)
    return _pixel_bounds(
        context, x, y, x + region_width, y + region_height, width, height)


def _apply_color_matrix(input_surface, matrix, bounds):
    """Apply a 4x5 color matrix, returning a new image surface."""
    width = input_surface.get_width()
    height = input_surface.get_height()
    output = _new_image_surface(width, height)
    stride = input_surface.get_stride()
    input_buffer = input_surface.get_data()
    output_buffer = output.get_data()
    left, top, right, bottom = bounds
    pack = struct.Struct('=BBBB').pack

    for y_pixel in range(top, bottom):
        row = y_pixel * stride
        for x_pixel in range(left, right):
            offset = row + x_pixel * 4
            b, g, r, a = input_buffer[offset:offset + 4]
            if a == 0:
                # Fully transparent pixels stay fully transparent, whatever
                # the matrix does with their unpremultiplied components.
                continue
            alpha = a / 255
            red = r / 255 / alpha
            green = g / 255 / alpha
            blue = b / 255 / alpha
            if matrix is LUMINANCE_TO_ALPHA_MATRIX:
                # The specification defines the luminance on the source
                # color only; the source alpha is preserved, RGB becoming 0.
                new_red = new_green = new_blue = 0.0
                new_alpha = (
                    LUMINANCE_RED * red + LUMINANCE_GREEN * green
                    + LUMINANCE_BLUE * blue) * alpha
            else:
                new_red = (
                    matrix[0] * red + matrix[1] * green + matrix[2] * blue
                    + matrix[3] * alpha + matrix[4])
                new_green = (
                    matrix[5] * red + matrix[6] * green + matrix[7] * blue
                    + matrix[8] * alpha + matrix[9])
                new_blue = (
                    matrix[10] * red + matrix[11] * green + matrix[12] * blue
                    + matrix[13] * alpha + matrix[14])
                new_alpha = (
                    matrix[15] * red + matrix[16] * green + matrix[17] * blue
                    + matrix[18] * alpha + matrix[19])
            new_red = min(1, max(0, new_red))
            new_green = min(1, max(0, new_green))
            new_blue = min(1, max(0, new_blue))
            new_alpha = min(1, max(0, new_alpha))
            output_buffer[offset:offset + 4] = pack(
                round(new_blue * new_alpha * 255),
                round(new_green * new_alpha * 255),
                round(new_red * new_alpha * 255),
                round(new_alpha * 255))
    output.mark_dirty()
    return output


def _combine_blend(input_surface, other_surface, mode, bounds):
    """Blend two surfaces with an SVG feBlend mode, clipped to ``bounds``."""
    output = _new_image_surface(
        input_surface.get_width(), input_surface.get_height())
    context = cairo.Context(output)
    context.set_source_surface(input_surface, 0, 0)
    context.paint()
    context.set_operator(
        BLEND_OPERATORS.get(mode, BLEND_OPERATORS['normal']))
    left, top, right, bottom = bounds
    context.rectangle(left, top, right - left, bottom - top)
    context.clip()
    context.set_source_surface(other_surface, 0, 0)
    context.paint()
    output.flush()
    return output


def _draw_flood(width, height, bounds, child, surface):
    """Build a transparent surface filled with the feFlood color."""
    output = _new_image_surface(width, height)
    context = cairo.Context(output)
    left, top, right, bottom = bounds
    context.rectangle(left, top, right - left, bottom - top)
    context.clip()
    _, paint_color = paint(child.get('flood-color'))
    context.set_source_rgba(*surface.map_color(
        paint_color, float(child.get('flood-opacity', 1))))
    context.paint()
    output.flush()
    return output


def _offset_surface(input_surface, dx, dy, bounds):
    """Translate a surface by device pixels, clipped to ``bounds``."""
    output = _new_image_surface(
        input_surface.get_width(), input_surface.get_height())
    context = cairo.Context(output)
    left, top, right, bottom = bounds
    context.rectangle(left, top, right - left, bottom - top)
    context.clip()
    context.set_source_surface(input_surface, dx, dy)
    context.paint()
    output.flush()
    return output


def apply_color_matrix_filter(surface, node, name):
    """Apply a filter containing pixel primitives to the current group.

    The painted content of the filtered node is available as the current
    group pattern. Return a surface pattern containing the filter result, or
    ``None`` when the filter can't be applied and the source graphic must be
    painted unchanged.

    """
    filter_node = surface.filters.get(name)
    if filter_node is None:
        return None

    context = surface.context
    group_pattern = context.pop_group()
    try:
        width = int(round(surface.width))
        height = int(round(surface.height))
        if width <= 0 or height <= 0:
            return group_pattern
        # SourceGraphic: the rasterized content of the filtered node.
        source_graphic = _rasterize_pattern(
            group_pattern, width, height)
        results = {'SourceGraphic': source_graphic}
        previous = source_graphic

        for child in filter_node.children:
            bounds = _primitive_subregion(
                surface, node, filter_node, child, width, height, context)
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                # Empty subregion: the primitive produces transparent black.
                output = _new_image_surface(width, height)
            elif child.tag == 'feColorMatrix':
                matrix = _parse_color_matrix(child)
                input_surface = _get_input(
                    child.get('in'), results, previous, source_graphic)
                output = _apply_color_matrix(
                    input_surface, matrix, bounds)
            elif child.tag == 'feOffset':
                input_surface = _get_input(
                    child.get('in'), results, previous, source_graphic)
                if filter_node.get('primitiveUnits') == 'objectBoundingBox':
                    bounding_box = calculate_bounding_box(surface, node)
                    if bounding_box and is_non_empty_bounding_box(
                            bounding_box):
                        _, _, box_width, box_height = bounding_box
                    else:
                        box_width = box_height = 0
                    dx = size(surface, child.get('dx', 0), 1) * box_width
                    dy = size(surface, child.get('dy', 0), 1) * box_height
                else:
                    dx = size(surface, child.get('dx', 0), 1)
                    dy = size(surface, child.get('dy', 0), 1)
                dx, dy = context.user_to_device_distance(dx, dy)
                output = _offset_surface(
                    input_surface, round(dx), round(dy), bounds)
            elif child.tag == 'feFlood':
                output = _draw_flood(width, height, bounds, child, surface)
            elif child.tag == 'feBlend':
                in_surface = _get_input(
                    child.get('in'), results, previous, source_graphic)
                in2_surface = _get_input(
                    child.get('in2'), results, previous, source_graphic)
                output = _combine_blend(
                    in_surface, in2_surface,
                    child.get('mode', 'normal'), bounds)
            else:
                # Unsupported primitives pass their input through unchanged.
                output = previous

            result_name = child.get('result')
            if result_name:
                results[result_name] = output
            previous = output

        # Clip the final result to the filter region.
        region = _filter_region(
            surface, node, filter_node, width, height, context)
        if (region[0] > 0 or region[1] > 0 or region[2] < width
                or region[3] < height):
            clipped = _new_image_surface(width, height)
            clip_context = cairo.Context(clipped)
            clip_context.rectangle(
                region[0], region[1],
                region[2] - region[0], region[3] - region[1])
            clip_context.clip()
            clip_context.set_source_surface(previous, 0, 0)
            clip_context.paint()
            clipped.flush()
            previous = clipped

        result_pattern = cairo.SurfacePattern(previous)
        return result_pattern
    except (ValueError, OverflowError, ArithmeticError, cairo.Error):
        # Invalid filter definition: paint the source graphic unchanged
        # instead of corrupting the whole canvas.
        return group_pattern


def _get_input(name, results, previous, source_graphic):
    """Resolve an ``in`` / ``in2`` attribute to an image surface."""
    if not name:
        # Missing in: previous primitive, or SourceGraphic for the first one.
        return previous
    if name == 'SourceGraphic':
        return source_graphic
    if name == 'SourceAlpha':
        output = cairo.ImageSurface(
            cairo.FORMAT_ARGB32, source_graphic.get_width(),
            source_graphic.get_height())
        context = cairo.Context(output)
        context.set_source_surface(source_graphic, 0, 0)
        context.paint()
        buffer = output.get_data()
        stride = output.get_stride()
        for row in range(output.get_height()):
            row_offset = row * stride
            for column in range(output.get_width()):
                offset = row_offset + column * 4
                # Keep the premultiplied alpha byte, zero the color channels.
                buffer[offset:offset + 3] = b'\x00\x00\x00'
        output.mark_dirty()
        return output
    if name in results:
        return results[name]
    # Unknown named input: transparent black, as required by the spec.
    return cairo.ImageSurface(
        cairo.FORMAT_ARGB32, source_graphic.get_width(),
        source_graphic.get_height())
