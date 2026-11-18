"""
SVG filter primitives.

This module implements a raster-based filter pipeline used when a filter
contains primitives that cannot be mapped directly on Cairo operators, such as
``feComposite`` (including its ``arithmetic`` operator).

Intermediate images are Cairo ARGB32 surfaces, i.e. *premultiplied* RGBA
pixels.  All the Porter-Duff compositing formulas given by the SVG
specification are defined on premultiplied colors, so every operation below
works directly on premultiplied values.  Pixels with a zero alpha are then
handled naturally, without any division by alpha and without the fringe
artifacts produced by unpremultiplied math.

See https://www.w3.org/TR/SVG11/filters.html and
https://www.w3.org/TR/filter-effects-1/.

"""

import math
import sys
import warnings

import numpy as np

from .bounding_box import calculate_bounding_box, is_non_empty_bounding_box
from .helpers import size
from .surface import cairo

# Cairo operators matching the feBlend modes (W3C compositing and blending).
_BLEND_OPERATORS = {
    'normal': cairo.OPERATOR_OVER,
    'multiply': cairo.OPERATOR_MULTIPLY,
    'screen': cairo.OPERATOR_SCREEN,
    'darken': cairo.OPERATOR_DARKEN,
    'lighten': cairo.OPERATOR_LIGHTEN,
}


def filter_uses_raster_pipeline(filter_node):
    """Check whether ``filter_node`` needs the raster filter pipeline.

    The raster pipeline is used as soon as the filter contains an
    ``feComposite`` primitive.  Filters mixing feComposite with the other
    supported primitives (feOffset, feFlood, feBlend) are also routed here so
    that primitives can be chained through their ``result`` / ``in``
    attributes.

    """
    return any(
        child.tag == 'feComposite' for child in filter_node.children)


# ---------------------------------------------------------------------------
# Image buffers
# ---------------------------------------------------------------------------

class _Image:
    """An RGBA filter image backed by a Cairo ARGB32 surface.

    The surface has the same device size as the filtered element's main
    surface; transparent pixels simply mean that the image does not cover this
    device location.

    """

    def __init__(self, surface):
        self.surface = surface

    @property
    def width(self):
        return self.surface.get_width()

    @property
    def height(self):
        return self.surface.get_height()

    def pixels(self):
        """Return the pixels as a premultiplied ``(height, width, 4)`` RGBA.

        A Cairo ARGB32 word stores alpha in its most significant byte and blue
        in its least significant.  Reading the native words as bytes yields the
        channels in memory order, whose position depends on endianness, so the
        byte order is selected explicitly instead of being assumed: BGRA on
        little-endian, ARGB on big-endian, reordered to RGBA.

        """
        self.surface.flush()
        words = self._argb_words().view(np.uint8).reshape(
            self.height, self.width, 4)
        if sys.byteorder == 'big':
            return words[:, :, (1, 2, 3, 0)].copy()
        return words[:, :, (2, 1, 0, 3)].copy()

    def write_pixels(self, pixels):
        """Write a premultiplied ``(height, width, 4)`` RGBA uint8 array."""
        rgba = np.ascontiguousarray(pixels, dtype=np.uint8).reshape(
            self.height, self.width, 4)
        # Rebuild ARGB words (alpha in the most significant byte).
        argb = np.zeros((self.height, self.width), dtype=np.uint32)
        argb |= rgba[:, :, 3].astype(np.uint32) << 24
        argb |= rgba[:, :, 0].astype(np.uint32) << 16
        argb |= rgba[:, :, 1].astype(np.uint32) << 8
        argb |= rgba[:, :, 2].astype(np.uint32)
        self._argb_words()[:] = argb
        self.surface.mark_dirty()

    def _argb_words(self):
        """View the surface as a ``(height, width)`` native uint32 array."""
        return np.frombuffer(
            self.surface.get_data(), dtype=np.uint32).reshape(
                self.height, self.width)

    def rgba(self):
        """Return premultiplied RGBA as a float64 array in ``[0, 1]``."""
        return self.pixels().astype(np.float64) / 255


def _new_image(width, height):
    """Create a transparent RGBA image of the given device size."""
    return _Image(cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height))


def _copy_image(image):
    """Duplicate an image (cairocffi has no surface-copy binding)."""
    new_image = _new_image(image.width, image.height)
    context = cairo.Context(new_image.surface)
    _paint_device(context, image.surface)
    return new_image


def _paint_device(context, surface):
    """Paint ``surface`` aligned to device pixels, ignoring the current CTM.

    A SurfacePattern with an identity matrix maps surface pixels onto context
    device pixels whatever the context transformation, which is what filter
    buffers need: their coordinates are already device coordinates.

    """
    pattern = cairo.SurfacePattern(surface)
    pattern.set_matrix(cairo.Matrix())
    context.set_source(pattern)
    context.paint()


def _clip_to_region(image, region, matrix):
    """Return ``image`` clipped to ``(x, y, width, height)`` in user space.

    Return the image unchanged when ``region`` is ``None``.  The clip path is
    built with the main context matrix, so rotated regions are handled exactly.

    """
    if region is None:
        return image
    x, y, width, height = region
    result = _new_image(image.width, image.height)
    context = cairo.Context(result.surface)
    context.set_matrix(matrix)
    context.rectangle(x, y, width, height)
    context.clip()
    context.set_matrix(cairo.Matrix())
    _paint_device(context, image.surface)
    return result


def _source_alpha(image):
    """Return the ``SourceAlpha`` of ``image``: black RGB, source alpha."""
    result = _new_image(image.width, image.height)
    rgba = image.pixels()
    out = np.zeros_like(rgba)
    out[:, :, 3] = rgba[:, :, 3]
    result.write_pixels(out)
    return result


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------

def _resolve_filter_region(surface, node, filter_node):
    """Return ``(x, y, width, height)`` of the filter region in user space.

    With the default ``objectBoundingBox`` filterUnits the default region is
    ``-10%, -10%, 120%, 120%`` of the filtered element's bounding box.

    """
    bounding_box = calculate_bounding_box(surface, node)
    if is_non_empty_bounding_box(bounding_box):
        box_x, box_y, box_width, box_height = bounding_box
    else:
        box_x = box_y = box_width = box_height = 0

    if filter_node.get('filterUnits') == 'userSpaceOnUse':
        return (
            size(surface, filter_node.get('x', 0), 'x'),
            size(surface, filter_node.get('y', 0), 'y'),
            size(surface, filter_node.get('width', '100%'), 'x'),
            size(surface, filter_node.get('height', '100%'), 'y'))

    def box_value(attribute, default, origin, box_size):
        # Values are fractions (or percentages) of the element bounding box;
        # size()'s percentage handling would wrongly use the viewport instead.
        raw = filter_node.get(attribute, default)
        if raw and raw.strip().endswith('%'):
            value = float(raw.strip()[:-1]) / 100
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = size(surface, raw, 'x') / box_size if box_size else 0
        return origin + value * box_size

    return (
        box_value('x', '-10%', box_x, box_width),
        box_value('y', '-10%', box_y, box_height),
        box_value('width', '120%', 0, box_width),
        box_value('height', '120%', 0, box_height))


def _primitive_region(surface, node, filter_node, primitive, default_region):
    """Resolve a primitive subregion in current user space."""
    if not any(key in primitive for key in ('x', 'y', 'width', 'height')):
        return default_region

    bounding_box = calculate_bounding_box(surface, node)
    if is_non_empty_bounding_box(bounding_box):
        box_x, box_y, box_width, box_height = bounding_box
    else:
        box_x = box_y = box_width = box_height = 0
    region_x, region_y, region_width, region_height = default_region

    if filter_node.get('primitiveUnits') == 'objectBoundingBox':
        return (
            box_x + size(surface, primitive.get('x', 0), 'x') * box_width,
            box_y + size(surface, primitive.get('y', 0), 'y') * box_height,
            size(surface, primitive.get('width', 1), 'x') * box_width,
            size(surface, primitive.get('height', 1), 'y') * box_height)

    # Default primitiveUnits: userSpaceOnUse.  x/y are absolute coordinates
    # in the current user space; unspecified attributes extend the region to
    # the corresponding edge of the filter region.
    return (
        size(surface, primitive.get('x'), 1) if 'x' in primitive
        else region_x,
        size(surface, primitive.get('y'), 1) if 'y' in primitive
        else region_y,
        size(surface, primitive.get('width'), 1) if 'width' in primitive
        else region_width,
        size(surface, primitive.get('height'), 1) if 'height' in primitive
        else region_height)


# ---------------------------------------------------------------------------
# feComposite
# ---------------------------------------------------------------------------

def _composite(in_image, in2_image, operator, coefficients):
    """Composite two premultiplied RGBA images according to ``operator``.

    ``in_image`` is the SVG source (``in``), ``in2_image`` the destination
    (``in2``).  Returns a new :class:`_Image`.

    """
    source = in_image.rgba()
    destination = in2_image.rgba()
    sr, sg, sb, sa = (source[:, :, i] for i in range(4))
    dr, dg, db, da = (destination[:, :, i] for i in range(4))

    if operator == 'over':
        # r = s + d * (1 - as)
        alpha = sa + da * (1 - sa)
        channels = (
            sr + dr * (1 - sa),
            sg + dg * (1 - sa),
            sb + db * (1 - sa))
    elif operator == 'in':
        # r = s * ad
        alpha = sa * da
        channels = (sr * da, sg * da, sb * da)
    elif operator == 'out':
        # r = s * (1 - ad)
        alpha = sa * (1 - da)
        channels = (sr * (1 - da), sg * (1 - da), sb * (1 - da))
    elif operator == 'atop':
        # r = s * ad + d * (1 - as)
        alpha = sa * da + da * (1 - sa)
        channels = (
            sr * da + dr * (1 - sa),
            sg * da + dg * (1 - sa),
            sb * da + db * (1 - sa))
    elif operator == 'xor':
        # r = s * (1 - ad) + d * (1 - as)
        alpha = sa * (1 - da) + da * (1 - sa)
        channels = (
            sr * (1 - da) + dr * (1 - sa),
            sg * (1 - da) + dg * (1 - sa),
            sb * (1 - da) + db * (1 - sa))
    else:
        # arithmetic:
        # r = k1 * d * s + k2 * s + k3 * d + k4
        k1, k2, k3, k4 = coefficients
        alpha = k1 * da * sa + k2 * sa + k3 * da + k4
        channels = tuple(
            k1 * d * s + k2 * s + k3 * d + k4
            for d, s in ((dr, sr), (dg, sg), (db, sb)))

    # Every channel, alpha included, is clamped to [0, 1] as required by the
    # specification (arithmetic may produce out-of-range values).
    rgba = np.empty((source.shape[0], source.shape[1], 4))
    rgba[:, :, 3] = np.clip(alpha, 0, 1)
    for i, channel in enumerate(channels):
        rgba[:, :, i] = np.clip(channel, 0, 1)

    return _image_from_rgba(rgba, in_image.width, in_image.height)


def _image_from_rgba(rgba, width, height):
    """Build an image from a premultiplied float RGBA ``[0, 1]`` array."""
    image = _new_image(width, height)
    rounded = np.rint(rgba * 255).astype(np.uint8)
    image.write_pixels(np.ascontiguousarray(rounded))
    return image


def _parse_coefficients(primitive):
    """Parse the arithmetic k1-k4 coefficients.

    Unparsable or non-finite values (NaN or infinity) are not valid numbers:
    they are replaced by 0 and a warning is emitted, instead of silently
    propagating NaN pixels.

    """
    coefficients = []
    for name in ('k1', 'k2', 'k3', 'k4'):
        raw = primitive.get(name, '0')
        try:
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError('not a finite number')
        except (TypeError, ValueError):
            warnings.warn(
                f'feComposite attribute {name}="{raw}" is not a finite '
                'number, using 0', RuntimeWarning, stacklevel=2)
            value = 0.0
        coefficients.append(value)
    return tuple(coefficients)


# ---------------------------------------------------------------------------
# Other primitives
# ---------------------------------------------------------------------------

def _offset_image(surface, node, filter_node, primitive, image):
    """Implement feOffset."""
    if filter_node.get('primitiveUnits') == 'objectBoundingBox':
        bounding_box = calculate_bounding_box(surface, node)
        if is_non_empty_bounding_box(bounding_box):
            _, _, box_width, box_height = bounding_box
        else:
            box_width = box_height = 0
        dx = size(surface, primitive.get('dx', 0), 1) * box_width
        dy = size(surface, primitive.get('dy', 0), 1) * box_height
    else:
        dx = size(surface, primitive.get('dx', 0), 1)
        dy = size(surface, primitive.get('dy', 0), 1)

    matrix = surface.context.get_matrix()
    dx_device = matrix.xx * dx + matrix.xy * dy
    dy_device = matrix.yx * dx + matrix.yy * dy

    result = _new_image(image.width, image.height)
    context = cairo.Context(result.surface)
    context.translate(dx_device, dy_device)
    _paint_device(context, image.surface)
    return result


def _flood_image(surface, primitive, width, height):
    """Implement feFlood: a solid color over the whole buffer.

    The primitive subregion clipping is applied afterwards by the pipeline.

    """
    image = _new_image(width, height)
    context = cairo.Context(image.surface)
    try:
        opacity = float(primitive.get('flood-opacity', 1))
    except (TypeError, ValueError):
        opacity = 1.0
    rgba = surface.map_color(
        primitive.get('flood-color', 'black'), opacity)
    context.set_source_rgba(*rgba)
    context.paint()
    return image


def _blend(in_image, in2_image, mode):
    """Implement feBlend using the matching Cairo blend operator."""
    if mode not in _BLEND_OPERATORS:
        warnings.warn(
            f'Unknown feBlend mode "{mode}", using "normal"',
            RuntimeWarning, stacklevel=2)
        mode = 'normal'
    result = _copy_image(in2_image)
    context = cairo.Context(result.surface)
    context.set_operator(_BLEND_OPERATORS[mode])
    _paint_device(context, in_image.surface)
    return result


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------

_UNSUPPORTED_INPUTS = frozenset((
    'BackgroundImage', 'BackgroundAlpha', 'FillPaint', 'StrokePaint'))


def _resolve_input(name, source_graphic, results, width, height):
    """Resolve an ``in`` / ``in2`` attribute to an :class:`_Image`."""
    if name in results:
        return results[name]
    if name == 'SourceGraphic':
        return source_graphic
    if name == 'SourceAlpha':
        return _source_alpha(source_graphic)
    if name in _UNSUPPORTED_INPUTS:
        # BackgroundImage and friends require an explicit enable-background
        # capture that CairoSVG does not implement: use transparent black.
        warnings.warn(
            f'Filter input "{name}" is not supported, '
            'using a transparent image', RuntimeWarning, stacklevel=2)
        return _new_image(width, height)
    if name:
        # An unknown reference is a broken input: transparent black.
        warnings.warn(
            f'Filter input "{name}" does not reference any previous result, '
            'using a transparent image', RuntimeWarning, stacklevel=2)
    return _new_image(width, height)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def _rasterize_source(surface, source_surface):
    """Return an ImageSurface holding ``source_surface``'s pixels.

    PNG output already uses an ImageSurface.  Vector outputs (PDF, PS, SVG)
    store the painted group in a RecordingSurface, which has no pixel access:
    replay it onto an image surface at the main surface device size.

    """
    if isinstance(source_surface, cairo.ImageSurface):
        return source_surface

    width = max(1, int(round(surface.width)))
    height = max(1, int(round(surface.height)))
    image_surface = cairo.ImageSurface(
        cairo.FORMAT_ARGB32, width, height)
    context = cairo.Context(image_surface)
    # The group was captured in device coordinates, so paint it without any
    # transformation.
    context.set_source_surface(source_surface, 0, 0)
    context.paint()
    return image_surface


def run_raster_filter(surface, node, filter_node, source_surface):
    """Compute the filter result.

    Return ``(result_image, matrix, filter_region)`` where ``result_image``
    is an :class:`_Image` the size of the main surface.  The filter output
    replaces SourceGraphic inside the filter region (including transparent
    result pixels); SourceGraphic is preserved outside the region.

    """
    matrix = surface.context.get_matrix()
    rasterized_source = _rasterize_source(surface, source_surface)
    width, height = (
        rasterized_source.get_width(), rasterized_source.get_height())
    source_graphic = _Image(rasterized_source)

    filter_region = _resolve_filter_region(surface, node, filter_node)
    region_x, region_y, region_width, region_height = filter_region
    if region_width <= 0 or region_height <= 0:
        # An invalid filter region disables the filter: keep the source.
        warnings.warn(
            'Filter has an empty region, the filter is not applied',
            RuntimeWarning, stacklevel=2)
        return source_graphic, matrix, filter_region

    results = {}
    previous = None
    primitive_index = 0
    for primitive in filter_node.children:
        tag = primitive.tag
        if tag not in ('feComposite', 'feOffset', 'feFlood', 'feBlend'):
            # Unknown primitives are skipped; the implicit input chain then
            # falls back to SourceGraphic or the previous real result.
            warnings.warn(
                f'Filter primitive "{tag}" is not supported, it is ignored',
                RuntimeWarning, stacklevel=2)
            continue

        region = _primitive_region(
            surface, node, filter_node, primitive, filter_region)

        if tag == 'feFlood':
            result = _flood_image(surface, primitive, width, height)
        else:
            in_name = primitive.get('in')
            if in_name is None:
                # The first primitive defaults to SourceGraphic; later ones
                # use the result of the previous primitive.
                input_image = (
                    source_graphic if primitive_index == 0 else previous)
                if input_image is None:
                    input_image = source_graphic
            else:
                input_image = _resolve_input(
                    in_name, source_graphic, results, width, height)

            if tag == 'feOffset':
                result = _offset_image(
                    surface, node, filter_node, primitive, input_image)
            elif tag == 'feBlend':
                in2_image = _resolve_input(
                    primitive.get('in2', 'SourceGraphic'),
                    source_graphic, results, width, height)
                result = _blend(
                    input_image, in2_image, primitive.get('mode', 'normal'))
            else:  # feComposite
                in2_image = _resolve_input(
                    primitive.get('in2', 'SourceGraphic'),
                    source_graphic, results, width, height)
                operator = primitive.get('operator', 'over')
                if operator == 'arithmetic':
                    coefficients = _parse_coefficients(primitive)
                elif operator not in (
                        'over', 'in', 'out', 'atop', 'xor'):
                    warnings.warn(
                        f'Unknown feComposite operator "{operator}", '
                        'using "over"', RuntimeWarning, stacklevel=2)
                    operator = 'over'
                    coefficients = (0, 0, 0, 0)
                else:
                    coefficients = (0, 0, 0, 0)
                result = _composite(
                    input_image, in2_image, operator, coefficients)

        # Each primitive result is clipped to its own subregion.
        result = _clip_to_region(result, region, matrix)

        result_name = primitive.get('result')
        if result_name:
            results[result_name] = result
        previous = result
        primitive_index += 1

    if previous is None:
        # No supported primitive produced anything: keep SourceGraphic.
        previous = source_graphic

    # Compose the final image: SourceGraphic outside the filter region, the
    # pipeline result inside it (even where the result is transparent).
    target = _new_image(width, height)
    target_context = cairo.Context(target.surface)
    _paint_device(target_context, rasterized_source)
    target_context.save()
    target_context.set_matrix(matrix)
    target_context.rectangle(
        region_x, region_y, region_width, region_height)
    target_context.clip()
    target_context.set_operator(cairo.OPERATOR_CLEAR)
    target_context.paint()
    target_context.set_operator(cairo.OPERATOR_OVER)
    target_context.restore()
    _paint_device(target_context, previous.surface)
    return target, matrix, filter_region


def apply_raster_filter(surface, node, filter_node, source_surface):
    """Run the raster filter pipeline and expose its result as the source.

    ``source_surface`` holds the filtered element's painted group, i.e. the
    ``SourceGraphic``.  The filtered image becomes the current context source
    without being painted yet, so that a mask or an opacity set on the same
    node applies to it exactly as for an unfiltered group.

    The pipeline works in device pixels, but the context current transform
    maps user space onto those device pixels.  The source pattern therefore
    carries that same matrix, otherwise the result would be misaligned as soon
    as the element is scaled or rotated.

    """
    result_image, matrix, _ = run_raster_filter(
        surface, node, filter_node, source_surface)
    pattern = cairo.SurfacePattern(result_image.surface)
    pattern.set_matrix(matrix)
    surface.context.set_source(pattern)
