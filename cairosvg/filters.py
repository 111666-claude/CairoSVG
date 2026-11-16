"""
SVG filter primitives.

This module implements a basic rendering pipeline for the filter primitives
that can not be drawn directly on the Cairo context and need to keep
intermediate results to combine them later, namely ``feComposite`` and the
primitives commonly chained with it (``feOffset``, ``feFlood`` and
``feBlend``).

Intermediate results are stored as Cairo groups: they keep the vector
information and use the same premultiplied-alpha color space as Cairo
itself, matching the SVG compositing rules.

"""

from .bounding_box import calculate_bounding_box, is_non_empty_bounding_box
from .helpers import paint, size
from .surface import cairo

#: Operators shared by ``feBlend`` and the basic Cairo compositing operators.
BLEND_OPERATORS = {
    'darken': cairo.OPERATOR_DARKEN,
    'lighten': cairo.OPERATOR_LIGHTEN,
    'multiply': cairo.OPERATOR_MULTIPLY,
    'normal': cairo.OPERATOR_OVER,
    'screen': cairo.OPERATOR_SCREEN,
}

#: ``feComposite`` operators mapped to the matching Cairo operators.
#:
#: Cairo operators take the source as the first operand and the backdrop as
#: the second one, which matches the SVG definition where the ``in`` image is
#: composited over the ``in2`` image. Both use premultiplied alpha.
COMPOSITE_OPERATORS = {
    'over': cairo.OPERATOR_OVER,
    'in': cairo.OPERATOR_IN,
    'out': cairo.OPERATOR_OUT,
    'atop': cairo.OPERATOR_ATOP,
    'xor': cairo.OPERATOR_XOR,
}

#: Filter primitives handled by the rendering pipeline.
PIPELINE_TAGS = frozenset((
    'feBlend', 'feComposite', 'feFlood', 'feOffset'))


def uses_pipeline(filter_node):
    """Check whether a filter needs the intermediate-result pipeline.

    Filters without ``feComposite`` keep using the legacy code path, so that
    their rendering is not changed.

    """
    return any(
        child.tag == 'feComposite' for child in filter_node.children)


def _object_bounding_box(surface, node):
    """Return a usable object bounding box for objectBoundingBox units."""
    bounding_box = calculate_bounding_box(surface, node)
    if is_non_empty_bounding_box(bounding_box):
        return bounding_box
    # The spec does not define what to do without a bounding box: use the
    # current viewport instead of dividing by zero.
    return (0, 0, surface.context_width, surface.context_height)


def _filter_region(surface, node, filter_node):
    """Return the filter region in user-space coordinates."""
    if filter_node.get('filterUnits') == 'userSpaceOnUse':
        x = size(surface, filter_node.get('x', '-10%'), 'x')
        y = size(surface, filter_node.get('y', '-10%'), 'y')
        width = size(surface, filter_node.get('width', '120%'), 'x')
        height = size(surface, filter_node.get('height', '120%'), 'y')
    else:
        bbox_x, bbox_y, bbox_width, bbox_height = _object_bounding_box(
            surface, node)
        x = bbox_x + size(
            surface, filter_node.get('x', '-10%'), bbox_width)
        y = bbox_y + size(
            surface, filter_node.get('y', '-10%'), bbox_height)
        width = size(
            surface, filter_node.get('width', '120%'), bbox_width)
        height = size(
            surface, filter_node.get('height', '120%'), bbox_height)
    return x, y, width, height


def _primitive_region(surface, node, filter_node, primitive):
    """Return the clipping subregion of a filter primitive."""
    if filter_node.get('primitiveUnits') == 'objectBoundingBox':
        bbox_x, bbox_y, bbox_width, bbox_height = _object_bounding_box(
            surface, node)
        # Fractions of the object bounding box: plain numbers and percentages
        # are both multiplied by the bounding box size.
        x = bbox_x + size(
            surface, primitive.get('x', '0'), 1) * bbox_width
        y = bbox_y + size(
            surface, primitive.get('y', '0'), 1) * bbox_height
        width = size(
            surface, primitive.get('width', '1'), 1) * bbox_width
        height = size(
            surface, primitive.get('height', '1'), 1) * bbox_height
    else:
        region_x, region_y, _, _ = _filter_region(
            surface, node, filter_node)
        x = region_x + size(surface, primitive.get('x', '0'), 'x')
        y = region_y + size(surface, primitive.get('y', '0'), 'y')
        width = size(surface, primitive.get('width', '100%'), 'x')
        height = size(surface, primitive.get('height', '100%'), 'y')
    return x, y, width, height


def _source_alpha(surface, source_graphic):
    """Return the alpha channel of the source graphic as a black image."""
    context = surface.context
    context.push_group()
    # Draw the source graphic, then replace its colors by opaque black kept
    # only where the graphic is opaque. Cairo stores premultiplied RGBA, so
    # the result is black with the source alpha channel.
    context.set_operator(cairo.OPERATOR_OVER)
    context.set_source(source_graphic)
    context.paint()
    context.set_source_rgba(0, 0, 0, 1)
    context.set_operator(cairo.OPERATOR_IN)
    context.paint()
    result = context.pop_group()
    context.set_operator(cairo.OPERATOR_OVER)
    return result


def _resolve_input(name, results, source_graphic, source_alpha, previous):
    """Resolve the ``in`` or ``in2`` reference of a filter primitive."""
    if name == 'SourceGraphic':
        return source_graphic
    if name == 'SourceAlpha':
        return source_alpha
    if name in results:
        return results[name]
    if name is None:
        # An omitted reference uses the previous primitive result, or the
        # source graphic for the first primitive.
        return previous if previous is not None else source_graphic
    # Missing or not-yet-defined references (including forward references and
    # result cycles) must not silently fall back to another image.
    raise ValueError(
        f'Abort rendering: filter input "{name}" is not defined')


def _new_group(surface, region):
    """Push a group clipped to the given user-space rectangle."""
    x, y, width, height = region
    surface.context.push_group()
    if width and height:
        surface.context.rectangle(x, y, width, height)
        surface.context.clip()


def _fe_offset(surface, node, filter_node, primitive, source, region):
    """Render an ``feOffset`` primitive."""
    if filter_node.get('primitiveUnits') == 'objectBoundingBox':
        _, _, bbox_width, bbox_height = _object_bounding_box(surface, node)
        dx = size(surface, primitive.get('dx', 0), 1) * bbox_width
        dy = size(surface, primitive.get('dy', 0), 1) * bbox_height
    else:
        dx = size(surface, primitive.get('dx', 0), 1)
        dy = size(surface, primitive.get('dy', 0), 1)
    _new_group(surface, region)
    context = surface.context
    context.translate(dx, dy)
    context.set_operator(cairo.OPERATOR_OVER)
    context.set_source(source)
    context.paint()
    return context.pop_group()


def _fe_flood(surface, primitive, region):
    """Render an ``feFlood`` primitive."""
    x, y, width, height = region
    _new_group(surface, region)
    context = surface.context
    _, flood_color = paint(primitive.get('flood-color'))
    context.set_source_rgba(*surface.map_color(
        flood_color or 'black',
        float(primitive.get('flood-opacity', 1))))
    context.set_operator(cairo.OPERATOR_OVER)
    context.rectangle(x, y, width, height)
    context.fill()
    return context.pop_group()


def _fe_blend(primitive, inputs, region, surface):
    """Render an ``feBlend`` primitive."""
    mode = primitive.get('mode', 'normal')
    if mode not in BLEND_OPERATORS:
        raise ValueError(
            f'Abort rendering: unsupported feBlend mode "{mode}"')
    _new_group(surface, region)
    context = surface.context
    context.set_operator(cairo.OPERATOR_OVER)
    context.set_source(inputs[1])
    context.paint()
    context.set_operator(BLEND_OPERATORS[mode])
    context.set_source(inputs[0])
    context.paint()
    return context.pop_group()


def _fe_composite(primitive, inputs, region, surface):
    """Render an ``feComposite`` primitive."""
    operator_name = primitive.get('operator', 'over')
    if operator_name not in COMPOSITE_OPERATORS:
        # Never degrade an unsupported arithmetic/Lighter operator to a
        # plain "over" compositing.
        raise ValueError(
            'Abort rendering: unsupported feComposite operator '
            f'"{operator_name}"')
    _new_group(surface, region)
    context = surface.context
    context.set_operator(cairo.OPERATOR_OVER)
    context.set_source(inputs[1])
    context.paint()
    context.set_operator(COMPOSITE_OPERATORS[operator_name])
    context.set_source(inputs[0])
    context.paint()
    return context.pop_group()


def apply_filter(surface, node, filter_node, source_graphic):
    """Run the filter primitives and return their final result.

    ``source_graphic`` is the Cairo group containing the painted element,
    returned by ``Context.pop_group()``. The returned pattern is clipped to
    the filter region when it is painted, leaving pixels outside of it
    untouched.

    """
    context = surface.context
    results = {}
    previous = None
    source_alpha = None

    for primitive in filter_node.children:
        if primitive.tag not in PIPELINE_TAGS:
            raise ValueError(
                'Abort rendering: unsupported filter primitive '
                f'"{primitive.tag}"')

        region = _primitive_region(surface, node, filter_node, primitive)

        if primitive.tag == 'feFlood':
            result = _fe_flood(surface, primitive, region)
        else:
            if source_alpha is None:
                source_alpha = _source_alpha(surface, source_graphic)
            if primitive.tag == 'feOffset':
                source = _resolve_input(
                    primitive.get('in'), results, source_graphic,
                    source_alpha, previous)
                result = _fe_offset(
                    surface, node, filter_node, primitive, source, region)
            else:
                inputs = [
                    _resolve_input(
                        primitive.get(attribute), results, source_graphic,
                        source_alpha, previous)
                    for attribute in ('in', 'in2')]
                if primitive.tag == 'feBlend':
                    result = _fe_blend(primitive, inputs, region, surface)
                else:
                    result = _fe_composite(
                        primitive, inputs, region, surface)

        previous = result
        result_name = primitive.get('result')
        if result_name:
            if result_name in results:
                # A result name defined more than once can only be part of a
                # cycle: references are only resolved against earlier
                # primitives.
                raise ValueError(
                    'Abort rendering: duplicated filter result '
                    f'"{result_name}"')
            results[result_name] = result

    if previous is None:
        previous = source_graphic

    # Clip the final result to the filter region so pixels outside of it are
    # never modified.
    context.save()
    context.rectangle(*_filter_region(surface, node, filter_node))
    context.clip()
    context.set_operator(cairo.OPERATOR_OVER)
    context.set_source(previous)
    context.paint()
    context.restore()
    return previous
