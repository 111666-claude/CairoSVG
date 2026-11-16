"""
SVG filter primitives.

This module implements the raster-based filter primitives that cannot be
expressed with native Cairo operations, currently ``feDisplacementMap``.

"""

import math
import sys

import numpy as np

from .bounding_box import calculate_bounding_box, is_non_empty_bounding_box
from .helpers import size
from .image import image
from .surface import cairo

# SVG 1.1 filter effects, section 15: the R and G channels are used by
# default for the horizontal and vertical displacement.
DEFAULT_X_CHANNEL = 'R'
DEFAULT_Y_CHANNEL = 'G'
CHANNEL_OFFSETS = {
    # Byte offset in a little-endian premultiplied ARGB32 buffer.
    'R': 2,
    'G': 1,
    'B': 0,
    'A': 3,
}

# Displacement samples are clamped to the source edges.  A displacement
# larger than this many pixels is treated as a malformed map instead of
# iterating over absurdly large coordinates.
MAX_DISPLACEMENT_PIXELS = 1_000_000

# Upper bound on the rasterized filter region, in device pixels.  It keeps a
# pathological region from allocating an unbounded amount of memory.
MAX_RASTER_PIXELS = 100_000_000


class FilterError(ValueError):
    """Raised when a filter primitive cannot be applied as requested."""


def apply_raster_filters(surface, node, filter_node):
    """Apply raster primitives while the filtered group is still active."""
    group_target = surface.context.get_group_target()
    if not isinstance(group_target, cairo.ImageSurface):
        # Non-raster targets (vector surfaces) are handled by the post-paint
        # hook once the group has been popped.
        return
    group_pattern = cairo.SurfacePattern(group_target)
    for child in filter_node.children:
        if child.tag == 'feDisplacementMap':
            apply_displacement_map(
                surface, node, filter_node, child,
                group_pattern=group_pattern)


def validate_displacement_map(surface, node, filter_node, primitive):
    """Validate an ``feDisplacementMap`` primitive, raising on errors.

    The validation happens before the filtered group is painted, so an
    invalid filter never silently behaves as if no filter was present.

    """
    in_name = primitive.get('in', 'SourceGraphic')
    in2_name = primitive.get('in2')
    if not in2_name:
        raise FilterError(
            'feDisplacementMap requires an "in2" displacement map')
    if in_name not in ('SourceGraphic', 'SourceAlpha'):
        raise FilterError(
            f'feDisplacementMap "in" input {in_name!r} is not supported, '
            'only "SourceGraphic" and "SourceAlpha" are')
    if in2_name not in ('SourceGraphic', 'SourceAlpha') and \
            in2_name not in surface.images:
        raise FilterError(
            f'feDisplacementMap "in2" input {in2_name!r} is not defined')

    x_channel = primitive.get('xChannelSelector', DEFAULT_X_CHANNEL)
    y_channel = primitive.get('yChannelSelector', DEFAULT_Y_CHANNEL)
    if x_channel not in CHANNEL_OFFSETS:
        raise FilterError(
            f'feDisplacementMap xChannelSelector {x_channel!r} is invalid, '
            'expected one of R, G, B or A')
    if y_channel not in CHANNEL_OFFSETS:
        raise FilterError(
            f'feDisplacementMap yChannelSelector {y_channel!r} is invalid, '
            'expected one of R, G, B or A')

    try:
        scale = float(primitive.get('scale', '0'))
    except (TypeError, ValueError):
        raise FilterError('feDisplacementMap "scale" must be a number')
    if not math.isfinite(scale):
        raise FilterError('feDisplacementMap "scale" must be a finite number')

    # objectBoundingBox units need a non-empty bounding box to resolve
    # coordinates and lengths.
    if filter_node.get('primitiveUnits') == 'objectBoundingBox':
        bounding_box = calculate_bounding_box(surface, node)
        if not is_non_empty_bounding_box(bounding_box):
            raise FilterError(
                'feDisplacementMap with primitiveUnits="objectBoundingBox" '
                'requires a non-empty object bounding box')


def _bounding_box_fraction(surface, value, default, box_length):
    """Resolve an objectBoundingBox length.

    Plain numbers are fractions of the bounding box side (``0.5`` = half
    of the box); percentages are divided by 100, as required by the SVG
    filter region specification.  :func:`helpers.size` already returns
    percentages as their decimal fraction when the reference is 1.

    """
    value = value if value is not None else default
    return size(surface, value, 1) * box_length


def _filter_region(surface, node, filter_node):
    """Return the filter region as ``(x, y, width, height)`` in user units."""
    if filter_node.get('filterUnits') == 'userSpaceOnUse':
        x = size(surface, filter_node.get('x', '-10%'), 'x')
        y = size(surface, filter_node.get('y', '-10%'), 'y')
        width = size(surface, filter_node.get('width', '120%'), 'x')
        height = size(surface, filter_node.get('height', '120%'), 'y')
    else:
        # Default filterUnits: objectBoundingBox.
        bounding_box = calculate_bounding_box(surface, node)
        if not is_non_empty_bounding_box(bounding_box):
            raise FilterError(
                'a filter with objectBoundingBox units requires a non-empty '
                'object bounding box')
        bx, by, bwidth, bheight = bounding_box
        x = bx + _bounding_box_fraction(
            surface, filter_node.get('x'), '-10%', bwidth)
        y = by + _bounding_box_fraction(
            surface, filter_node.get('y'), '-10%', bheight)
        width = _bounding_box_fraction(
            surface, filter_node.get('width'), '120%', bwidth)
        height = _bounding_box_fraction(
            surface, filter_node.get('height'), '120%', bheight)

    if not (math.isfinite(x) and math.isfinite(y) and
            math.isfinite(width) and math.isfinite(height)) or \
            width <= 0 or height <= 0:
        raise FilterError('the filter region must be a finite non-empty rect')
    return x, y, width, height


def _primitive_region(surface, node, filter_node, primitive, filter_region):
    """Return the primitive subregion in user units."""
    if primitive.get('x') is None and primitive.get('y') is None and \
            primitive.get('width') is None and primitive.get('height') \
            is None:
        return filter_region

    if filter_node.get('primitiveUnits') == 'objectBoundingBox':
        bounding_box = calculate_bounding_box(surface, node)
        bx, by, bwidth, bheight = bounding_box
        x = bx + _bounding_box_fraction(
            surface, primitive.get('x'), '0', bwidth)
        y = by + _bounding_box_fraction(
            surface, primitive.get('y'), '0', bheight)
        width = _bounding_box_fraction(
            surface, primitive.get('width'), '1', bwidth)
        height = _bounding_box_fraction(
            surface, primitive.get('height'), '1', bheight)
    else:
        x = size(surface, primitive.get('x', '0'), 'x')
        y = size(surface, primitive.get('y', '0'), 'y')
        width = size(surface, primitive.get('width', '1'), 'x')
        height = size(surface, primitive.get('height', '1'), 'y')
    if not (math.isfinite(x) and math.isfinite(y) and
            math.isfinite(width) and math.isfinite(height)) or \
            width <= 0 or height <= 0:
        raise FilterError(
            'the feDisplacementMap subregion must be a finite non-empty rect')
    return x, y, width, height


def _device_region(surface, region):
    """Map a user-space rectangle to a whole device-space pixel box.

    Return ``(origin_x, origin_y, width, height)``.  The box is not clipped
    to the target surface: a filter region may extend past the surface
    edges, and keeping the unclamped box preserves the 1:1 mapping between
    local raster pixels and device coordinates.  Use
    :func:`_clamped_device_region` when writing pixels onto the target.

    """
    x, y, width, height = region
    ctm = surface.context.get_matrix()
    points = (
        ctm.transform_point(x, y),
        ctm.transform_point(x + width, y),
        ctm.transform_point(x, y + height),
        ctm.transform_point(x + width, y + height),
    )
    min_x = math.floor(min(point[0] for point in points))
    min_y = math.floor(min(point[1] for point in points))
    max_x = math.ceil(max(point[0] for point in points))
    max_y = math.ceil(max(point[1] for point in points))
    return min_x, min_y, max_x - min_x, max_y - min_y


def _clamped_device_region(surface, device_region):
    """Intersect a device pixel box with the target surface.

    Return ``(target_x, target_y, local_x, local_y, width, height)`` where
    the first two coordinates address the target surface and the local
    coordinates address the raster box, or ``None`` when the boxes do not
    overlap.

    """
    origin_x, origin_y, width, height = device_region
    target_x0 = max(0, origin_x)
    target_y0 = max(0, origin_y)
    target_x1 = min(surface.width, origin_x + width)
    target_y1 = min(surface.height, origin_y + height)
    if target_x1 <= target_x0 or target_y1 <= target_y0:
        return None
    return (
        target_x0, target_y0,
        target_x0 - origin_x, target_y0 - origin_y,
        target_x1 - target_x0, target_y1 - target_y0)


def _read_surface(image_surface):
    """Return an ``(height, width, 4)`` uint8 view of an ARGB32 surface."""
    width = image_surface.get_width()
    height = image_surface.get_height()
    stride = image_surface.get_stride()
    array = np.frombuffer(
        image_surface.get_data(), dtype=np.uint8).reshape(height, stride)
    return np.ascontiguousarray(array[:, :width * 4]).reshape(
        height, width, 4)


def _rasterize_pattern(context, pattern, device_region):
    """Paint a Cairo pattern on a new surface, aligned on device pixels."""
    origin_x, origin_y, width, height = device_region
    image_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    new_context = cairo.Context(image_surface)

    # A pushed group stores its content in device space.  Translating the
    # pattern by the box origin aligns its pixels 1:1 with the new surface
    # regardless of the user-space CTM in effect when the group was built.
    matrix = cairo.Matrix(1, 0, 0, 1, origin_x, origin_y)
    saved_matrix = pattern.get_matrix()
    pattern.set_matrix(matrix)
    new_context.set_source(pattern)
    new_context.paint()
    pattern.set_matrix(saved_matrix)
    image_surface.flush()
    return image_surface


class _ImageSurfaceShim:
    """Minimal surface-like object expected by :func:`cairosvg.image.image`."""


def _rasterize_image_node(surface, node, device_region):
    """Render an ``feImage`` node aligned on the target device pixels."""
    origin_x, origin_y, width, height = device_region
    image_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    new_context = cairo.Context(image_surface)

    ctm = surface.context.get_matrix()
    # Keep the same user->device transform but offset it so that device
    # pixel (origin_x, origin_y) of the target becomes pixel (0, 0) of this
    # raster box; the feImage geometry stays expressed in user coordinates.
    matrix = ctm.copy()
    matrix.x0 -= origin_x
    matrix.y0 -= origin_y
    new_context.set_matrix(matrix)

    shim = _ImageSurfaceShim()
    shim.context = new_context
    shim.dpi = surface.dpi
    shim.context_width = surface.context_width
    shim.context_height = surface.context_height
    shim.font_size = surface.font_size
    shim.map_color = surface.map_color
    shim.map_image = surface.map_image
    shim.tree_cache = surface.tree_cache
    shim.images = surface.images

    def draw(child):
        # Nested SVG images are drawn with the same coordinate transforms.
        saved_matrix = surface.context.get_matrix()
        surface.context.save()
        surface.context.set_matrix(new_context.get_matrix())
        try:
            surface.draw(child)
        finally:
            surface.context.restore()
            surface.context.set_matrix(saved_matrix)

    shim.draw = draw
    # The generic image() drawer recognizes raster nodes by their tag.
    saved_tag = node.tag
    node.tag = 'image'
    # A displacement map is sampled per pixel: force nearest-neighbor so a
    # constant map is not altered by bilinear filtering at its edges.
    saved_rendering = node.get('image-rendering')
    node['image-rendering'] = 'pixelated'
    try:
        image(shim, node)
    finally:
        node.tag = saved_tag
        if saved_rendering is None:
            node.pop('image-rendering', None)
        else:
            node['image-rendering'] = saved_rendering
    image_surface.flush()
    return image_surface


def _source_alpha(array):
    """Return the SourceGraphic alpha channel as an opaque gray image."""
    alpha = array[:, :, 3]
    result = np.zeros_like(array)
    result[:, :, 0] = alpha
    result[:, :, 1] = alpha
    result[:, :, 2] = alpha
    result[:, :, 3] = 255
    return result


def apply_displacement_map(surface, node, filter_node, primitive,
                           group_pattern=None):
    """Apply an ``feDisplacementMap`` primitive to the current source."""
    if sys.byteorder != 'little':
        raise FilterError(
            'feDisplacementMap is only supported on little-endian systems')

    context = surface.context
    if group_pattern is None:
        group_pattern = context.get_source()
    if not isinstance(group_pattern, cairo.SurfacePattern):
        raise FilterError(
            'feDisplacementMap can only be applied to a painted group')

    scale = float(primitive.get('scale', '0'))
    if scale == 0.0:
        # A null displacement is the identity operation, pixel for pixel.
        return

    filter_region = _filter_region(surface, node, filter_node)
    region = _primitive_region(
        surface, node, filter_node, primitive, filter_region)
    device_region = _device_region(surface, region)
    clamped = _clamped_device_region(surface, device_region)
    if clamped is None:
        # The primitive region lies fully outside the target surface.
        return
    origin_x, origin_y, pixel_width, pixel_height = device_region
    if pixel_width * pixel_height > MAX_RASTER_PIXELS:
        raise FilterError(
            'feDisplacementMap region is too large to rasterize '
            f'({pixel_width}x{pixel_height} pixels)')

    source_surface = _rasterize_pattern(
        context, group_pattern, device_region)
    source = _read_surface(source_surface)

    in2_name = primitive.get('in2')
    if in2_name in ('SourceGraphic', 'SourceAlpha'):
        map_array = (
            _source_alpha(source) if in2_name == 'SourceAlpha' else source)
    else:
        map_node = surface.images[in2_name]
        if map_node.tag != 'feImage':
            raise FilterError(
                f'feDisplacementMap "in2" input {in2_name!r} must be an '
                'feImage element')
        map_surface = _rasterize_image_node(
            surface, map_node, device_region)
        map_array = _read_surface(map_surface)

    in_name = primitive.get('in', 'SourceGraphic')
    if in_name == 'SourceAlpha':
        source = _source_alpha(source)

    x_channel = primitive.get('xChannelSelector', DEFAULT_X_CHANNEL)
    y_channel = primitive.get('yChannelSelector', DEFAULT_Y_CHANNEL)
    x_offset = CHANNEL_OFFSETS[x_channel]
    y_offset = CHANNEL_OFFSETS[y_channel]

    # The displacement scale is expressed in user units.  The current CTM
    # turns user units into device (pixel) units.
    ctm = context.get_matrix()
    user_scale_x = math.hypot(ctm.xx, ctm.xy)
    user_scale_y = math.hypot(ctm.yx, ctm.yy)
    if abs(scale) * max(user_scale_x, user_scale_y) > \
            MAX_DISPLACEMENT_PIXELS:
        raise FilterError(
            'feDisplacementMap scale leads to a displacement larger than '
            f'{MAX_DISPLACEMENT_PIXELS} pixels')

    # SVG channel values give signed displacements in [-0.5, 0.5].
    x_map = map_array[:, :, x_offset].astype(np.float32) / 255 - 0.5
    y_map = map_array[:, :, y_offset].astype(np.float32) / 255 - 0.5
    displacement_x = scale * x_map * user_scale_x
    displacement_y = scale * y_map * user_scale_y
    if not (np.isfinite(displacement_x).all() and
            np.isfinite(displacement_y).all()):
        raise FilterError(
            'feDisplacementMap displacement map contains non-finite values')

    columns, rows = np.meshgrid(
        np.arange(pixel_width, dtype=np.float32),
        np.arange(pixel_height, dtype=np.float32))
    # Inverse mapping: the value written at (x, y) is sampled from the
    # source at (x - displacement, y - displacement), as required by the
    # SVG specification.
    sample_x = columns - displacement_x
    sample_y = rows - displacement_y

    # Clamp the continuous coordinates before interpolating: a sample
    # outside the source must take its value entirely from the nearest
    # edge pixel (rather than mixing it with a duplicated neighbor).
    sample_x = np.clip(sample_x, 0, pixel_width - 1)
    sample_y = np.clip(sample_y, 0, pixel_height - 1)
    floor_x = np.floor(sample_x).astype(np.int64)
    floor_y = np.floor(sample_y).astype(np.int64)
    frac_x = (sample_x - floor_x).astype(np.float32)
    frac_y = (sample_y - floor_y).astype(np.float32)
    x0 = floor_x
    x1 = np.clip(floor_x + 1, 0, pixel_width - 1)
    y0 = floor_y
    y1 = np.clip(floor_y + 1, 0, pixel_height - 1)

    # Interpolation runs on premultiplied values, which keeps fully
    # transparent pixels black instead of bleeding color through them.
    top = source[y0, x0].astype(np.float32) * (1 - frac_x[:, :, None]) + \
        source[y0, x1].astype(np.float32) * frac_x[:, :, None]
    bottom = source[y1, x0].astype(np.float32) * \
        (1 - frac_x[:, :, None]) + \
        source[y1, x1].astype(np.float32) * frac_x[:, :, None]
    sampled = top * (1 - frac_y[:, :, None]) + \
        bottom * frac_y[:, :, None]
    output = np.rint(sampled).astype(np.uint8)

    result_surface = cairo.ImageSurface(
        cairo.FORMAT_ARGB32, pixel_width, pixel_height)
    result_stride = result_surface.get_stride()
    result_buffer = np.frombuffer(
        result_surface.get_data(), dtype=np.uint8).reshape(
        pixel_height, result_stride)
    # ImageSurface contents are not zero-initialized; clear padding and
    # untouched pixels before writing the computed rows.
    result_buffer.fill(0)
    result_buffer[:, :pixel_width * 4] = output.reshape(
        pixel_height, pixel_width * 4)
    result_surface.mark_dirty()

    # Write the displaced pixels back at the exact device pixels they came
    # from.  For raster targets the buffer is edited directly, which bypasses
    # any active clip and compositing operator and guarantees that pixels
    # outside the box are untouched.  Only the part intersecting the target
    # surface is copied.
    (target_x, target_y, local_x, local_y,
     clamped_width, clamped_height) = clamped
    # The filter runs before pop_group, so the current group target is the
    # live surface the group content will be read from when it is composited
    # onto the parent target.  Writing into it directly replaces the source
    # pixels within the filter region.
    target = context.get_group_target()
    if isinstance(target, cairo.ImageSurface):
        target_stride = target.get_stride()
        target_buffer = np.frombuffer(
            target.get_data(), dtype=np.uint8).reshape(
            target.get_height(), target_stride)
        target_buffer[
            target_y:target_y + clamped_height,
            target_x * 4:target_x * 4 + clamped_width * 4] = \
            output[
                local_y:local_y + clamped_height,
                local_x:local_x + clamped_width].reshape(
                clamped_height, clamped_width * 4)
        target.mark_dirty()
        return

    # Vector and recording surfaces: fall back to a clipped Cairo paint.
    result_pattern = cairo.SurfacePattern(result_surface)
    device_to_user = ctm.copy()
    device_to_user.invert()
    pattern_matrix = device_to_user.multiply(
        cairo.Matrix(1, 0, 0, 1, origin_x, origin_y))
    result_pattern.set_matrix(pattern_matrix)
    context.save()
    context.set_source(result_pattern)
    context.set_operator(cairo.OPERATOR_SOURCE)
    context.new_path()
    user_x, user_y = device_to_user.transform_point(target_x, target_y)
    user_x2, user_y2 = device_to_user.transform_point(
        target_x + clamped_width, target_y + clamped_height)
    context.rectangle(
        user_x, user_y, user_x2 - user_x, user_y2 - user_y)
    context.fill()
    context.restore()
    context.set_matrix(ctm)
