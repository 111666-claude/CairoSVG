"""
Optional collection of painted paths and point hit-testing.

This module implements the machinery behind the ``path_collector`` parameter
of the ``svg2*`` functions. When a :class:`PathCollector` instance is passed,
every path actually painted on the main canvas (from ``path`` elements, basic
shapes and ``use``-generated content) is recorded in draw order, with
coordinates already transformed into output device units (pixels for PNG).
When no collector is passed, rendering is strictly unchanged.

"""

import cairocffi as cairo

_PATH_OPS = {
    cairo.PATH_MOVE_TO: 'move',
    cairo.PATH_LINE_TO: 'line',
    cairo.PATH_CURVE_TO: 'curve',
    cairo.PATH_CLOSE_PATH: 'close',
}

# Tolerance used to approximate curves with line segments for hit-testing,
# in output device units.
FLATTENING_TOLERANCE = 0.1


def _context_segments(context):
    """Return the context's current path as transformed, normalized segments.

    The returned list contains tuples:

    - ``('move', (x, y))``
    - ``('line', (x, y))``
    - ``('curve', ((x1, y1), (x2, y2), (x3, y3)))``
    - ``('close',)``

    Coordinates are transformed by the context's current matrix, and
    degenerate moves (not followed by a drawn segment) are dropped.

    """
    matrix = context.get_matrix()
    segments = []
    pending_move = None
    for op_code, points in context.copy_path():
        op = _PATH_OPS[op_code]
        if op == 'move':
            # Delay moves so that degenerate ones (empty subpaths, including
            # the move cairo adds after each close) can be dropped.
            pending_move = matrix.transform_point(*points)
            continue
        if pending_move is not None:
            segments.append(('move', pending_move))
            pending_move = None
        if op == 'line':
            segments.append(('line', matrix.transform_point(*points)))
        elif op == 'curve':
            segments.append(('curve', tuple(
                matrix.transform_point(*points[i:i + 2])
                for i in (0, 2, 4))))
        else:  # close
            segments.append(('close',))
    return tuple(segments)


def _rect_segments(x, y, width, height, matrix):
    """Return the segments of a rectangle transformed by ``matrix``."""
    corners = [matrix.transform_point(x, y)]
    corners.append(matrix.transform_point(x + width, y))
    corners.append(matrix.transform_point(x + width, y + height))
    corners.append(matrix.transform_point(x, y + height))
    return (
        ('move', corners[0]),
        ('line', corners[1]),
        ('line', corners[2]),
        ('line', corners[3]),
        ('close',))


def _flatten_cubic(p0, p1, p2, p3, output, depth=0):
    """Approximate a cubic Bézier with line segments appended to ``output``."""
    if depth < 16:
        # Flatness: distance of the control points from the p0-p3 chord
        chord_x = p3[0] - p0[0]
        chord_y = p3[1] - p0[1]
        chord_length = (chord_x ** 2 + chord_y ** 2) ** .5
        flat = True
        for control in (p1, p2):
            if chord_length:
                distance = abs(
                    chord_x * (p0[1] - control[1]) -
                    (p0[0] - control[0]) * chord_y) / chord_length
            else:
                distance = (
                    (control[0] - p0[0]) ** 2 +
                    (control[1] - p0[1]) ** 2) ** .5
            if distance > FLATTENING_TOLERANCE:
                flat = False
                break
        if flat:
            output.append(p3)
            return
        # Split the curve in two (de Casteljau) and flatten both halves
        p01 = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        p12 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        p23 = ((p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2)
        p012 = ((p01[0] + p12[0]) / 2, (p01[1] + p12[1]) / 2)
        p123 = ((p12[0] + p23[0]) / 2, (p12[1] + p23[1]) / 2)
        p0123 = ((p012[0] + p123[0]) / 2, (p012[1] + p123[1]) / 2)
        _flatten_cubic(p0, p01, p012, p0123, output, depth + 1)
        _flatten_cubic(p0123, p123, p23, p3, output, depth + 1)
    else:
        output.append(p3)


def _flatten(segments):
    """Split ``segments`` into subpaths of points, approximating curves."""
    subpaths = []
    subpath = None
    position = None
    for segment in segments:
        op = segment[0]
        if op == 'move':
            if subpath:
                subpaths.append(subpath)
            subpath = [segment[1]]
            position = segment[1]
        elif op == 'line':
            if subpath is None:
                subpath = [] if position is None else [position]
            subpath.append(segment[1])
            position = segment[1]
        elif op == 'curve':
            if subpath is None:
                subpath = [] if position is None else [position]
            _flatten_cubic(position, *segment[1], subpath)
            position = segment[1][2]
    if subpath:
        subpaths.append(subpath)
    return subpaths


def _is_left(x1, y1, x2, y2, x, y):
    """Return a positive value if ``(x, y)`` is left of the (1 -> 2) line."""
    return (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)


def _point_in_subpaths(subpaths, x, y, fill_rule):
    """Test whether ``(x, y)`` is inside ``subpaths`` given ``fill_rule``.

    Subpaths are implicitly closed, as cairo does when filling.

    """
    winding = 0
    crossings = 0
    for subpath in subpaths:
        count = len(subpath)
        if count < 2:
            continue
        for i in range(count):
            x1, y1 = subpath[i]
            x2, y2 = subpath[(i + 1) % count]
            if y1 <= y:
                if y2 > y:
                    if _is_left(x1, y1, x2, y2, x, y) > 0:
                        winding += 1
                    if x1 + (y - y1) * (x2 - x1) / (y2 - y1) > x:
                        crossings += 1
            elif y2 <= y:
                if _is_left(x1, y1, x2, y2, x, y) < 0:
                    winding -= 1
                if x1 + (y - y1) * (x2 - x1) / (y2 - y1) > x:
                    crossings += 1
    if fill_rule == 'evenodd':
        return crossings % 2 == 1
    return winding != 0


class CollectedPath:
    """A path recorded while rendering, in draw order.

    Attributes:
    - ``id``: the ``id`` attribute of the source element (``None`` if unset).
      For content instantiated by ``use``, this is the id of the referenced
      element.
    - ``tag``: the tag of the source element (``path``, ``rect``, …).
    - ``segments``: the path's segments (see :func:`_context_segments`),
      transformed into output device units.
    - ``closed``: whether the path contains a close operation.
    - ``fill_rule``: the element's ``fill-rule`` (``nonzero`` or ``evenodd``).

    """

    def __init__(self, node_id, tag, segments, closed, fill_rule, clips):
        """Create a collected path; used by :class:`PathCollector`."""
        self.id = node_id
        self.tag = tag
        self.segments = segments
        self.closed = closed
        self.fill_rule = fill_rule
        self._clips = clips
        self._flat = None
        self._flat_clips = None

    def __repr__(self):
        """Return a debug representation of the collected path."""
        return (
            f'<CollectedPath id={self.id!r} tag={self.tag!r} '
            f'segments={len(self.segments)} closed={self.closed}>')

    def contains(self, x, y, fill_rule=None):
        """Test whether the filled region of the path contains ``(x, y)``.

        ``fill_rule`` is ``nonzero`` or ``evenodd``; when ``None``, the
        element's own ``fill-rule`` is used. The clips that were active when
        the path was painted (viewport, ``clip`` and ``clip-path``) are taken
        into account: a point outside them is never inside the path.

        """
        if self._flat is None:
            self._flat = _flatten(self.segments)
        if not _point_in_subpaths(
                self._flat, x, y, fill_rule or self.fill_rule):
            return False
        if self._flat_clips is None:
            self._flat_clips = [_flatten(clip) for clip in self._clips]
        return all(
            _point_in_subpaths(clip, x, y, 'nonzero')
            for clip in self._flat_clips)


class PathCollector:
    """Collect the paths painted while rendering an SVG document.

    Pass an instance as the ``path_collector`` parameter of one of the
    ``svg2*`` functions. After rendering, ``paths`` contains one
    :class:`CollectedPath` per painted path, in draw order.

    """

    def __init__(self):
        """Create an empty collector, ready to be passed to ``svg2*``."""
        self.paths = []
        self._clips = []

    def hit(self, x, y, fill_rule=None):
        """Return the collected paths whose fill region contains ``(x, y)``.

        The result is a list of :class:`CollectedPath` in draw order.
        ``fill_rule`` (``nonzero`` or ``evenodd``) overrides each element's
        own ``fill-rule`` when given.

        """
        return [
            path for path in self.paths if path.contains(x, y, fill_rule)]

    # The methods below are hooks used by ``surface.Surface`` while drawing.

    def _collect(self, node, context):
        """Record the context's current path as painted by ``node``."""
        segments = _context_segments(context)
        self.paths.append(CollectedPath(
            node.get('id'), node.tag, segments,
            closed=any(segment[0] == 'close' for segment in segments),
            fill_rule=node.get('fill-rule', 'nonzero'),
            clips=tuple(self._clips)))

    def _push_clip_path(self, context):
        """Record the context's current path as an active clip."""
        self._clips.append(_context_segments(context))

    def _push_clip_rect(self, x, y, width, height, matrix):
        """Record a transformed rectangle as an active clip."""
        self._clips.append(_rect_segments(x, y, width, height, matrix))

    def _clip_depth(self):
        """Return the current number of active clips."""
        return len(self._clips)

    def _truncate_clips(self, depth):
        """Forget the clips pushed after ``depth`` was recorded."""
        del self._clips[depth:]
