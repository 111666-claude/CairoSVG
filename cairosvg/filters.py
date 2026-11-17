"""
SVG filter primitives.

This module implements the pixel-based filter primitives that cannot be
expressed directly with Cairo vector operations, such as
``feComponentTransfer``.

"""

import math
import sys

try:
    import numpy
except ImportError:  # pragma: no cover
    numpy = None

from .surface import cairo

#: The four channels handled by ``feComponentTransfer``, mapped to the channel
#: index used in the RGBA arrays below (0=R, 1=G, 2=B, 3=A).
CHANNELS = ('R', 'G', 'B', 'A')

#: Last-resort transfer description, leaving a channel unchanged.
IDENTITY = ('identity',)

#: Native byte order, used to decode Cairo ARGB32 32-bit words.
LITTLE_ENDIAN = sys.byteorder == 'little'


def _finite_numbers(values):
    """Parse a whitespace/comma separated list of finite floats.

    Return ``None`` when any token is missing, not a number or not finite.

    """
    numbers = []
    for value in values.replace(',', ' ').split():
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        numbers.append(number)
    return numbers


def _parse_channel(node):
    """Parse one ``feFuncX`` element into a normalized transfer description.

    The first item of the returned tuple is one of ``identity``, ``table``,
    ``discrete``, ``linear`` or ``gamma``. Invalid or non-finite parameters
    make the channel fall back to the identity transfer, giving malformed
    documents a deterministic rendering.

    """
    type_ = node.get('type', 'identity').strip().lower()

    if type_ == 'identity':
        return IDENTITY

    if type_ == 'linear':
        slope = _finite_numbers(node.get('slope', '1'))
        intercept = _finite_numbers(node.get('intercept', '0'))
        if slope is None or intercept is None:
            return IDENTITY
        return ('linear', slope[0], intercept[0])

    if type_ == 'gamma':
        amplitude = _finite_numbers(node.get('amplitude', '1'))
        exponent = _finite_numbers(node.get('exponent', '1'))
        offset = _finite_numbers(node.get('offset', '0'))
        if (amplitude is None or exponent is None or offset is None or
                exponent[0] <= 0):
            # The exponent must be a strictly positive finite number.
            return IDENTITY
        return ('gamma', amplitude[0], exponent[0], offset[0])

    if type_ in ('table', 'discrete'):
        values = _finite_numbers(node.get('tableValues', ''))
        if not values:
            # An empty or invalid lookup table leaves the channel unchanged.
            return IDENTITY
        return (type_, values)

    # Unknown type: leave the channel unchanged.
    return IDENTITY


def parse_component_transfer(filter_node):
    """Parse the ``feFuncR/G/B/A`` children of a ``feComponentTransfer``.

    Return a dict mapping each channel index (0=R, 1=G, 2=B, 3=A) to a
    transfer description. Unspecified channels keep their original value and
    repeated definitions of a given channel follow the last one, as required
    by the specification.

    """
    transfers = {index: IDENTITY for index in range(4)}
    for child in filter_node.children:
        tag = child.tag.lower()
        # The parser keeps lowercase function tags as ``fefuncR`` but
        # uppercase ones as ``funcR``; accept both spellings.
        suffix = tag[2:] if tag.startswith('fe') else tag
        if suffix.startswith('func') and suffix[-1] in 'rgba':
            transfers[CHANNELS.index(suffix[-1].upper())] = \
                _parse_channel(child)
    return transfers


def _clamp(value):
    """Clamp a scalar to the legal ``[0, 1]`` range, mapping NaN/inf to 0."""
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def _lookup(value, transfer):
    """Apply a ``table`` or ``discrete`` transfer to a scalar value."""
    type_, table = transfer
    count = len(table)
    if type_ == 'table':
        position = value * (count - 1)
        lower = int(position)
        if lower >= count - 1:
            return table[-1]
        fraction = position - lower
        return table[lower] + fraction * (table[lower + 1] - table[lower])
    entry = int(value * count)
    if entry >= count:
        entry = count - 1
    return table[entry]


def _apply_scalar(transfers, pixels):
    """Apply ``transfers`` to a flat ARGB32 byte buffer (pure Python).

    Transfer indices are RGBA (0=R, 1=G, 2=B, 3=A). Each channel is mapped to
    its byte position inside the native 32-bit word (``0xAARRGGBB``).

    """
    # Byte offset of each RGBA channel inside a native ARGB32 word.
    if LITTLE_ENDIAN:
        rgba_byte = (2, 1, 0, 3)
    else:  # pragma: no cover
        rgba_byte = (1, 2, 3, 0)

    for index in range(0, len(pixels), 4):
        positions = [index + rgba_byte[i] for i in range(4)]
        alpha = pixels[positions[3]]
        # Read the straight (non-premultiplied) RGBA values.
        channels = [0.0, 0.0, 0.0, alpha / 255.0]
        if alpha:
            scale = 255.0 / alpha
            for rgba_index in range(3):
                value = pixels[positions[rgba_index]] * scale / 255.0
                channels[rgba_index] = min(1.0, max(0.0, value))

        for rgba_index, transfer in transfers.items():
            type_ = transfer[0]
            value = channels[rgba_index]
            if type_ == 'identity':
                continue
            if type_ == 'linear':
                value = transfer[1] * value + transfer[2]
            elif type_ == 'gamma':
                _, amplitude, exponent, offset = transfer
                value = amplitude * value ** exponent + offset
            else:
                value = _lookup(value, transfer)
            channels[rgba_index] = _clamp(value)

        # Re-premultiply the color components with the mapped alpha.
        new_alpha = channels[3]
        for rgba_index in range(3):
            value = (
                channels[rgba_index] * new_alpha * 255.0
                if new_alpha else 0.0)
            pixels[positions[rgba_index]] = int(round(value))
        pixels[positions[3]] = int(round(new_alpha * 255.0))


def _map_numpy(values, transfer):
    """Apply one transfer description to a NumPy array of channel values."""
    type_ = transfer[0]
    if type_ == 'identity':
        return values
    if type_ == 'linear':
        values = transfer[1] * values + transfer[2]
    elif type_ == 'gamma':
        _, amplitude, exponent, offset = transfer
        values = amplitude * values ** exponent + offset
    else:
        table = numpy.asarray(transfer[1], dtype=numpy.float64)
        count = len(table)
        if type_ == 'table':
            position = values * (count - 1)
            lower = numpy.floor(position).astype(numpy.int64)
            fraction = position - numpy.floor(position)
            low = numpy.clip(lower, 0, count - 1)
            high = numpy.clip(lower + 1, 0, count - 1)
            values = table[low] + fraction * (table[high] - table[low])
        else:
            entry = numpy.floor(values * count).astype(numpy.int64)
            numpy.clip(entry, 0, count - 1, out=entry)
            values = table[entry]
    # Overflows and invalid operations become 0 before clamping to [0, 1].
    values = numpy.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return numpy.clip(values, 0.0, 1.0)


def _apply_numpy(transfers, data, width, height, stride):
    """Apply ``transfers`` to an ARGB32 buffer using NumPy.

    Cairo packs each pixel as a native 32-bit word ``0xAARRGGBB``. The
    working array indexes channels as 0=B, 1=G, 2=R, 3=A to match the low to
    high byte order of the word. The RGBA indices of ``transfers``
    (0=R, 1=G, 2=B, 3=A) are remapped to that memory layout.

    """
    # Working-array channel index for each RGBA channel (R, G, B, A).
    # The array follows the low-to-high byte order of the 0xAARRGGBB word:
    # index 0 is the low byte (B) and index 2 is the third byte (R).
    rgba_to_memory = (2, 1, 0, 3)

    words = numpy.frombuffer(
        data, dtype=numpy.uint32, count=height * stride // 4)
    words = words.reshape(height, stride // 4)[:, :width].copy()

    channels = numpy.empty((height, width, 4), dtype=numpy.float64)
    if LITTLE_ENDIAN:
        # Pixel word packed as 0xAARRGGBB.
        channels[..., 0] = words & 0xff          # B (low byte)
        channels[..., 1] = (words >> 8) & 0xff   # G
        channels[..., 2] = (words >> 16) & 0xff  # R
        channels[..., 3] = (words >> 24) & 0xff  # A (high byte)
    else:  # pragma: no cover
        channels[..., 0] = (words >> 24) & 0xff  # B
        channels[..., 1] = (words >> 16) & 0xff  # G
        channels[..., 2] = (words >> 8) & 0xff   # R
        channels[..., 3] = words & 0xff          # A

    alpha = channels[..., 3]
    nonzero = alpha > 0
    # Component transfer operates on non-premultiplied color values.
    for channel_index in range(3):
        unpremultiplied = numpy.zeros_like(alpha)
        numpy.divide(
            channels[..., channel_index] * 255.0, alpha,
            out=unpremultiplied, where=nonzero)
        channels[..., channel_index] = unpremultiplied / 255.0
    channels[..., 3] = alpha / 255.0

    with numpy.errstate(all='ignore'):
        for rgba_index, transfer in transfers.items():
            memory_index = rgba_to_memory[rgba_index]
            channels[..., memory_index] = _map_numpy(
                channels[..., memory_index], transfer)

    # Re-premultiply the color components with the (possibly mapped) alpha.
    # ``colors`` follows the memory channel order (B, G, R).
    new_alpha = channels[..., 3]
    colors = [
        numpy.rint(channels[..., i] * new_alpha * 255.0).astype(numpy.uint32)
        for i in range(3)]
    alpha_bytes = numpy.rint(new_alpha * 255.0).astype(numpy.uint32)

    if LITTLE_ENDIAN:
        # Native 0xAARRGGBB word: B in the low byte, R in the third byte.
        result = (
            colors[0] | (colors[1] << 8) | (colors[2] << 16) |
            (alpha_bytes << 24))
    else:  # pragma: no cover
        result = (
            (alpha_bytes << 24) | (colors[0] << 16) |
            (colors[1] << 8) | colors[2])

    view = numpy.frombuffer(
        data, dtype=numpy.uint32, count=height * stride // 4
    ).reshape(height, stride // 4)
    view[:, :width] = result


def _copy_image_surface(source):
    """Return an independent ARGB32 copy of an image ``source`` surface."""
    width, height = source.get_width(), source.get_height()
    copy = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    copy.get_data()[:] = bytes(source.get_data())
    copy.mark_dirty()
    return copy


def component_transfer(filter_node, source):
    """Apply an ``feComponentTransfer`` primitive to ``source``.

    A new image surface with the same dimensions and filter region as the
    input is returned; the input surface itself is never modified, so other
    primitives referencing the same input keep their original pixels. The
    transfer functions are applied to non-premultiplied color values and the
    alpha is restored afterwards.

    """
    transfers = parse_component_transfer(filter_node)

    if (not hasattr(source, 'get_format') or
            source.get_format() != cairo.FORMAT_ARGB32):
        # Recording surfaces (vector backends such as PDF and PS) expose no
        # pixel buffer: rasterize them first, keeping the same region as the
        # input reference.
        x, y, region_width, region_height = source.ink_extents()
        width = max(1, int(math.ceil(x + region_width)))
        height = max(1, int(math.ceil(y + region_height)))
        raster = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        context = cairo.Context(raster)
        context.set_source_surface(source, 0, 0)
        context.paint()
        source = raster

    target = _copy_image_surface(source)
    width, height = target.get_width(), target.get_height()
    stride = target.get_stride()
    buffer = target.get_data()

    if numpy is not None:
        _apply_numpy(transfers, buffer, width, height, stride)
    else:  # pragma: no cover
        pixels = list(bytes(buffer))
        _apply_scalar(transfers, pixels)
        buffer[:] = bytes(pixels)

    target.mark_dirty()
    return target
