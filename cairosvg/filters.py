"""SVG filter primitives.

This module implements basic support of the ``feTile`` filter primitive.
The reference tile is built from the effective region of ``SourceGraphic`` or
``SourceAlpha`` and repeated over the filter primitive subregion, keeping the
tile origin aligned with the SVG user coordinate system.

Only standard ``SourceGraphic`` and ``SourceAlpha`` inputs can be tiled:
intermediate results and the ``BackgroundImage`` / ``FillPaint`` /
``StrokePaint`` inputs are not supported and are reported explicitly instead
of being silently ignored.

"""

from math import ceil, floor, hypot, isfinite, sqrt

from .bounding_box import calculate_bounding_box, is_non_empty_bounding_box
from .helpers import size
from .surface import cairo

#: Maximum number of visible tiles painted by a single ``feTile``. Cairo
#: repeats patterns lazily (constant time per output pixel), so this is only a
#: sanity guard against absurd values: the actual memory cost is bounded by
#: ``MAX_TILE_PIXELS`` and by the output surface size.
MAX_TILE_REPEATS = 10 ** 12

#: Maximum number of pixels of the intermediate tile image (64 MiB).
MAX_TILE_PIXELS = 16_000_000

#: Filter inputs supported by this basic ``feTile`` implementation.
SUPPORTED_INPUTS = ('SourceGraphic', 'SourceAlpha')


def has_tile(filter_node):
    """Whether ``filter_node`` contains an ``feTile`` primitive.

    Filters containing ``feTile`` are handled by this module. Compositions
    that are not supported (for example an ``feTile`` that is not the last
    primitive, or a non-standard input) raise an explicit error instead of
    silently producing a wrong image.

    """
    return any(child.tag == 'feTile' for child in filter_node.children)


def _element_bounds(surface, node):
    """Return the ``(x, y, width, height)`` bounds of ``node``.

    Bounds are expressed in the current user space (the node transform has
    already been applied to the Cairo context) and are only returned when they
    have a finite, positive size. They define the reference tile region.

    """
    try:
        bounding_box = calculate_bounding_box(surface, node)
    except (TypeError, ValueError):
        bounding_box = None
    if is_non_empty_bounding_box(bounding_box):
        if all(isfinite(value) for value in bounding_box):
            return bounding_box

    # Shapes without a bounding box method (e.g. images) and nodes whose
    # children cannot be measured fall back to their own geometry.
    x = size(surface, node.get('x'), 'x')
    y = size(surface, node.get('y'), 'y')
    width = size(surface, node.get('width'), 'x')
    height = size(surface, node.get('height'), 'y')
    if (width > 0 and height > 0 and
            all(isfinite(value) for value in (x, y, width, height))):
        return x, y, width, height
    return None


def _object_bounding_box(surface, node):
    """Return the object bounding box, falling back to the viewport."""
    bounds = _element_bounds(surface, node)
    if bounds is None:
        # The specification does not define what to do without a bounding
        # box: use the viewport instead of dividing by zero.
        return 0, 0, surface.context_width, surface.context_height
    return bounds


def _object_bounding_box_length(surface, value, reference):
    """Resolve a length in object bounding box units.

    In ``objectBoundingBox`` units, plain numbers and percentages both
    represent fractions of the bounding box size.

    """
    if value is None:
        return 0
    if isinstance(value, str) and value.strip().endswith('%'):
        return size(surface, value, reference)
    return size(surface, value, 1) * reference


def _filter_region(surface, node, filter_node):
    """Return the filter region in user-space coordinates."""
    if filter_node.get('filterUnits') == 'userSpaceOnUse':
        x = size(surface, filter_node.get('x', '-10%'), 'x')
        y = size(surface, filter_node.get('y', '-10%'), 'y')
        width = size(surface, filter_node.get('width', '120%'), 'x')
        height = size(surface, filter_node.get('height', '120%'), 'y')
    else:
        bounding_x, bounding_y, bounding_width, bounding_height = (
            _object_bounding_box(surface, node))
        x = bounding_x + _object_bounding_box_length(
            surface, filter_node.get('x', '-10%'), bounding_width)
        y = bounding_y + _object_bounding_box_length(
            surface, filter_node.get('y', '-10%'), bounding_height)
        width = _object_bounding_box_length(
            surface, filter_node.get('width', '120%'), bounding_width)
        height = _object_bounding_box_length(
            surface, filter_node.get('height', '120%'), bounding_height)
    return x, y, width, height


def _primitive_subregion(surface, node, filter_node, primitive, region):
    """Return the subregion of a filter primitive in user space.

    Missing attributes default to the filter region, except in
    ``objectBoundingBox`` units where they are fractions of the object
    bounding box, as required by the specification.

    """
    region_x, region_y, region_width, region_height = region
    if filter_node.get('primitiveUnits') == 'objectBoundingBox':
        bounding_x, bounding_y, bounding_width, bounding_height = (
            _object_bounding_box(surface, node))
        x = (
            bounding_x + _object_bounding_box_length(
                surface, primitive['x'], bounding_width)
            if 'x' in primitive else region_x)
        y = (
            bounding_y + _object_bounding_box_length(
                surface, primitive['y'], bounding_height)
            if 'y' in primitive else region_y)
        width = (
            _object_bounding_box_length(
                surface, primitive['width'], bounding_width)
            if 'width' in primitive else region_width)
        height = (
            _object_bounding_box_length(
                surface, primitive['height'], bounding_height)
            if 'height' in primitive else region_height)
    else:
        # In user-space units, percentage lengths of filter primitives are
        # relative to the filter region dimensions, not to the viewport.
        x = (
            size(surface, primitive['x'], region_width)
            if 'x' in primitive else region_x)
        y = (
            size(surface, primitive['y'], region_height)
            if 'y' in primitive else region_y)
        width = (
            size(surface, primitive['width'], region_width)
            if 'width' in primitive else region_width)
        height = (
            size(surface, primitive['height'], region_height)
            if 'height' in primitive else region_height)
    return x, y, width, height


def _intersection(first, second):
    """Return the intersection of two rectangles, or ``None`` if empty."""
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    x = max(first_x, second_x)
    y = max(first_y, second_y)
    right = min(first_x + first_width, second_x + second_width)
    bottom = min(first_y + first_height, second_y + second_height)
    if right <= x or bottom <= y:
        return None
    return x, y, right - x, bottom - y


def _valid_rectangle(rectangle):
    """Whether a ``(x, y, width, height)`` rectangle can be rendered."""
    return (
        rectangle is not None and
        all(isfinite(value) for value in rectangle) and
        rectangle[2] > 0 and rectangle[3] > 0)


def _check_rectangle(rectangle, name):
    """Validate a filter rectangle, raising on invalid definitions.

    Negative sizes are errors according to the specification, while zero-sized
    rectangles simply disable the filter. Non-finite values can not be
    rendered and must not be used to allocate surfaces.

    """
    if not all(isfinite(value) for value in rectangle):
        raise ValueError(f'Abort rendering: {name} is not finite')
    if rectangle[2] < 0 or rectangle[3] < 0:
        raise ValueError(f'Abort rendering: {name} has a negative size')


def _visible_region(surface, rectangle):
    """Clip a user-space rectangle to the visible part of the surface.

    Return ``None`` when the rectangle is fully outside of the surface.

    """
    matrix = surface.context.get_matrix().copy()
    try:
        matrix.invert()
    except cairo.Error:
        return rectangle
    corners = [
        matrix.transform_point(x, y)
        for x, y in (
            (0, 0), (surface.width, 0),
            (0, surface.height), (surface.width, surface.height))]
    xs = [corner[0] for corner in corners]
    ys = [corner[1] for corner in corners]
    viewport = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
    if not all(isfinite(value) for value in viewport):
        return rectangle
    return _intersection(rectangle, viewport)


def _tile_primitive(filter_node):
    """Return the ``feTile`` primitive whose result is the filter output.

    Other primitives are accepted before ``feTile``; their results are only
    ignored because the filter output is the result of the last primitive and
    ``feTile`` can only reference a standard input.

    """
    primitives = [
        child for child in filter_node.children
        if child.tag.startswith('fe')]
    if not primitives or primitives[-1].tag != 'feTile':
        raise ValueError(
            'Abort rendering: feTile is only supported as the last filter '
            'primitive of a filter')
    return primitives, primitives[-1]


def _tile_input(primitive, has_previous):
    """Return the validated input name of the ``feTile`` primitive."""
    input_name = primitive.get('in')
    if input_name is None:
        if has_previous:
            raise ValueError(
                'Abort rendering: feTile without an explicit input after '
                'another filter primitive is not supported')
        # The first primitive of a filter uses SourceGraphic by default.
        input_name = 'SourceGraphic'
    if input_name not in SUPPORTED_INPUTS:
        raise ValueError(
            f'Abort rendering: unsupported feTile input "{input_name}"')
    return input_name


def _tile_pixel_size(context, width, height):
    """Return the device-pixel size used to sample the tile."""
    matrix = context.get_matrix()
    scale_x = hypot(matrix.xx, matrix.yx)
    scale_y = hypot(matrix.xy, matrix.yy)
    if not (isfinite(scale_x) and isfinite(scale_y)):
        raise ValueError('Abort rendering: invalid filter transformation')
    # Degenerate transformations are rendered at one pixel per user unit.
    scale_x = scale_x if scale_x > 0 else 1
    scale_y = scale_y if scale_y > 0 else 1
    pixel_width = width * scale_x
    pixel_height = height * scale_y
    if not (isfinite(pixel_width) and isfinite(pixel_height)):
        return None
    # A tile smaller than a device pixel is sampled as a single pixel, keeping
    # the average color of the tile.
    pixel_width = max(1, ceil(pixel_width - 1e-9))
    pixel_height = max(1, ceil(pixel_height - 1e-9))

    # Bound the memory used by the intermediate tile image, degrading the
    # resolution rather than allocating an unbounded surface.
    if pixel_width * pixel_height > MAX_TILE_PIXELS:
        factor = sqrt(MAX_TILE_PIXELS / (pixel_width * pixel_height))
        pixel_width = max(1, floor(pixel_width * factor))
        pixel_height = max(1, floor(pixel_height * factor))
    return pixel_width, pixel_height


def tile_pattern(surface, node, filter_node, source):
    """Build a repeating pattern for an ``feTile`` filter.

    ``source`` is the Cairo group pattern returned by ``pop_group`` for the
    painted element. Return ``(pattern, clip)`` where ``pattern`` is ``None``
    when the tile is empty (the filter result is transparent), and ``clip`` is
    the user-space rectangle covered by the filter primitive (``None`` when
    nothing may be painted).

    """
    primitives, primitive = _tile_primitive(filter_node)
    input_name = _tile_input(primitive, len(primitives) > 1)

    region = _filter_region(surface, node, filter_node)
    _check_rectangle(region, 'filter region')
    if region[2] == 0 or region[3] == 0:
        # A zero-sized filter region disables rendering of the element.
        return None, None

    subregion = _primitive_subregion(
        surface, node, filter_node, primitive, region)
    _check_rectangle(subregion, 'feTile primitive subregion')
    if subregion[2] == 0 or subregion[3] == 0:
        # A zero-sized primitive subregion makes the result transparent.
        return None, None

    clip = _intersection(subregion, region)
    if clip is None:
        return None, None

    tile = _element_bounds(surface, node)
    if tile is None or not _valid_rectangle(tile):
        # Empty tile (no geometry at all): the filter result is fully
        # transparent.
        return None, clip
    _check_rectangle(tile, 'feTile input region')

    # The filter region is a hard clip on the filter primitive input: a tile
    # entirely outside of it produces a transparent result.
    if _intersection(tile, region) is None:
        return None, clip

    # Count only the tiles that can actually be seen, so that a huge filter
    # region around a small viewport does not trigger the repetition limit.
    visible = _visible_region(surface, clip)
    if visible is None:
        return None, clip
    repeats_x = ceil(visible[2] / tile[2]) + 1
    repeats_y = ceil(visible[3] / tile[3]) + 1
    if repeats_x * repeats_y > MAX_TILE_REPEATS:
        raise ValueError(
            'Abort rendering: too many feTile repetitions '
            f'({repeats_x} x {repeats_y})')

    pixel_size = _tile_pixel_size(surface.context, tile[2], tile[3])
    if pixel_size is None:
        return None, clip
    pixel_width, pixel_height = pixel_size
    scale_x = pixel_width / tile[2]
    scale_y = pixel_height / tile[3]

    # Map user space to tile pixels: the tile origin is the top-left corner
    # of the input image's effective region, so that tiles are anchored at
    # (x + i * width, y + j * height) in SVG user coordinates.
    matrix = cairo.Matrix(
        scale_x, 0, 0, scale_y, -tile[0] * scale_x, -tile[1] * scale_y)

    tile_surface = cairo.ImageSurface(
        cairo.FORMAT_ARGB32, pixel_width, pixel_height)
    tile_context = cairo.Context(tile_surface)
    tile_context.set_matrix(matrix)
    # The filter region is a hard clip applied to the input image too: parts
    # of the element outside of it are not part of the reference tile.
    tile_context.rectangle(*region)
    tile_context.clip()
    tile_context.set_source(source)
    tile_context.paint()
    if input_name == 'SourceAlpha':
        # Keep the alpha channel of the source and make its color black.
        tile_context.set_operator(cairo.OPERATOR_IN)
        tile_context.set_source_rgba(0, 0, 0, 1)
        tile_context.paint()
    tile_surface.flush()

    pattern = cairo.SurfacePattern(tile_surface)
    pattern.set_matrix(matrix)
    pattern.set_extend(cairo.EXTEND_REPEAT)
    return pattern, clip
