"""
Fonts managers.

CSS ``@font-face`` rules referencing TTF/OTF fonts are resolved here. The font
bytes are fetched with the same URL fetcher as every other external resource,
so the security policy chosen by the caller (``unsafe`` or a custom
``url_fetcher``) is never bypassed.

Loaded fonts are registered as cairo FreeType font faces: they are not
installed on the system and are therefore invisible to every other process or
library. The cairo FreeType backend keeps a global cache of the scaled fonts
built from a memory face and reads the font memory again on each glyph
request, so the font bytes have to remain valid for the whole process. They
are therefore stored in a process-level cache keyed by the font content. The
:class:`FontLibrary` of a single conversion only keeps track of the rules and
of the faces it uses: nothing about a document (its aliases, its URLs or its
security context) ever leaks to another conversion.

"""

import ctypes
import ctypes.util
import hashlib
import logging
import os

import tinycss2

from .url import parse_url

# Formats accepted in the ``format()`` hint of an ``src`` descriptor.
# WOFF and WOFF2 are compressed containers that FreeType may not be built to
# load directly, so they are skipped and the next source is tried.
SUPPORTED_FORMATS = frozenset((
    'truetype', 'opentype', 'ttf', 'otf', 'embedded-opentype'))

# File extensions used when the ``src`` descriptor gives no ``format()`` hint.
SUPPORTED_EXTENSIONS = frozenset(('.ttf', '.otf'))

# ``font-weight`` values considered bold, matching the threshold used when
# mapping the SVG attribute to a cairo font weight.
BOLD_THRESHOLD = 550

logger = logging.getLogger('cairosvg')

# The FreeType library and the cairo-ft entry point are loaded lazily. They are
# unavailable on minimal installations of cairo built without FreeType: in
# that case ``@font-face`` fonts are simply ignored and the historical system
# font behavior is kept.
_ft_library = None
_ft_library_ref = None
_cairo_create_font_face = None
_cairo_font_face_status = None
_load_attempted = False

# Process-level cache of the loaded fonts, keyed by the SHA-256 of the font
# bytes. Each entry keeps alive everything cairo needs to keep using the face:
# the font bytes, the FreeType face and the cairo font face wrapper.
#
# The same physical font referenced by several documents (or by several
# ``@font-face`` rules with different aliases) is therefore loaded once.
_loaded_fonts = {}


class FontRule:
    """A single CSS ``@font-face`` rule with a loadable TTF/OTF source."""

    def __init__(self, family, slant, weight, url):
        self.family = family
        # cairo slant/weight constants, as used by ``select_font_face``
        self.slant = slant
        self.weight = weight
        self.url = url


class LoadedFont:
    """A font loaded once and shared by the conversions that need it."""

    def __init__(self, font_buffer, font_size, ft_face, font_face):
        # The cairo FreeType backend may read this memory again as long as the
        # font is used, even after the conversion that loaded it has ended:
        # the buffer is therefore owned by the process-level cache.
        self.font_buffer = font_buffer
        self.font_size = font_size
        self.ft_face = ft_face
        self.font_face = font_face


def _load_backend():
    """Load the FreeType library and the cairo-ft constructor once.

    Return :data:`True` when the backend is usable, :data:`False` otherwise.

    """
    global _ft_library, _ft_library_ref, _cairo_create_font_face
    global _cairo_font_face_status, _load_attempted
    if _load_attempted:
        return _ft_library is not None
    _load_attempted = True
    freetype_path = ctypes.util.find_library('freetype')
    cairo_path = ctypes.util.find_library('cairo')
    if not freetype_path or not cairo_path:
        logger.warning(
            'Cannot find FreeType or cairo, falling back to system fonts')
        return False
    try:
        freetype = ctypes.CDLL(freetype_path)
        freetype.FT_Init_FreeType.argtypes = [
            ctypes.POINTER(ctypes.c_void_p)]
        freetype.FT_New_Memory_Face.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_long,
            ctypes.c_long, ctypes.POINTER(ctypes.c_void_p)]
        freetype.FT_Done_Face.argtypes = [ctypes.c_void_p]

        cairo_library = ctypes.CDLL(cairo_path)
        create = cairo_library.cairo_ft_font_face_create_for_ft_face
        create.argtypes = [ctypes.c_void_p, ctypes.c_int]
        create.restype = ctypes.c_void_p
        cairo_library.cairo_font_face_status.argtypes = [ctypes.c_void_p]
        cairo_library.cairo_font_face_status.restype = ctypes.c_int
    except (OSError, TypeError, AttributeError) as exception:
        logger.warning(
            'Cannot load @font-face fonts, falling back to system fonts: %s',
            exception)
        return False

    library = ctypes.c_void_p()
    if freetype.FT_Init_FreeType(ctypes.byref(library)):
        logger.warning(
            'Cannot initialize FreeType, falling back to system fonts')
        return False
    _ft_library = freetype
    _ft_library_ref = library
    _cairo_create_font_face = create
    _cairo_font_face_status = cairo_library.cairo_font_face_status
    return True


def parse_font_face_rules(tree):
    """Find the ``@font-face`` rules of the stylesheets of ``tree``.

    Only rules with at least one supported TTF/OTF URL source are returned.
    Relative URLs are resolved against the document URL, exactly as for images
    or CSS imports.

    """
    # Imported here to avoid a circular import at module load time.
    from .css import find_stylesheets
    from .surface import cairo

    rules = []
    for stylesheet in find_stylesheets(tree):
        for rule in stylesheet:
            if (rule.type != 'at-rule' or rule.content is None or
                    rule.lower_at_keyword != 'font-face'):
                continue
            font_rule = _parse_font_face_rule(tree, rule, cairo)
            if font_rule is not None:
                rules.append(font_rule)
    return rules


def _parse_font_face_rule(tree, rule, cairo):
    """Convert a tinycss2 ``@font-face`` rule into a :class:`FontRule`."""
    family = None
    style = 'normal'
    weight = 'normal'
    src = None
    for declaration in tinycss2.parse_declaration_list(
            rule.content, skip_comments=True, skip_whitespace=True):
        if declaration.type != 'declaration':
            continue
        if declaration.lower_name == 'font-family':
            values = [
                token.value for token in declaration.value
                if token.type in ('string', 'ident')]
            if values:
                family = values[0].strip('"\' ')
        elif declaration.lower_name == 'font-style':
            values = [
                token.value for token in declaration.value
                if token.type == 'ident']
            if values:
                style = values[0]
        elif declaration.lower_name == 'font-weight':
            values = [
                token.value for token in declaration.value
                if token.type in ('ident', 'number')]
            if values:
                # Numeric weights arrive as floats (e.g. 700.0), keywords as
                # strings such as "bold".
                weight = (
                    str(int(values[0])) if isinstance(values[0], (int, float))
                    else str(values[0]))
        elif declaration.lower_name == 'src':
            src = declaration.value
    if not family or not src:
        return None

    url = _find_font_url(src, tree)
    if url is None:
        logger.warning(
            '@font-face rule for "%s" has no supported TTF/OTF source, '
            'using system fonts instead', family)
        return None

    slant = (
        cairo.FONT_SLANT_ITALIC if style in ('italic', 'oblique')
        else cairo.FONT_SLANT_NORMAL)
    if weight in ('bold', 'bolder'):
        bold = True
    elif weight.isdigit():
        bold = int(weight) >= BOLD_THRESHOLD
    else:
        bold = False
    font_weight = (
        cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    return FontRule(family, slant, font_weight, url)


def _find_font_url(src_tokens, tree):
    """Return the first supported TTF/OTF URL found in a ``src`` value."""
    # Split the token list on commas: each group is one font source.
    groups = [[]]
    for token in src_tokens:
        if token.type == 'literal' and token.value == ',':
            groups.append([])
        else:
            groups[-1].append(token)

    for group in groups:
        url_token = next(
            (token for token in group if token.type == 'url'), None)
        if url_token is None:
            # ``local()`` sources rely on fonts already installed on the
            # system: they are handled by cairo itself, skip them.
            continue
        format_hint = _format_hint(group)
        if format_hint is not None and format_hint not in SUPPORTED_FORMATS:
            continue
        if format_hint is None:
            extension = os.path.splitext(
                parse_url(url_token.value).path)[1].lower()
            if extension and extension not in SUPPORTED_EXTENSIONS:
                continue
        return parse_url(url_token.value, tree.url)
    return None


def _format_hint(group):
    """Extract the lowercase ``format()`` hint of a ``src`` group."""
    for token in group:
        if token.type == 'function' and token.lower_name == 'format':
            for argument in token.arguments:
                if argument.type == 'string':
                    return argument.value.lower()
    return None


class FontLibrary:
    """The ``@font-face`` fonts available during a single conversion."""

    def __init__(self, tree):
        # Rules hold document-specific aliases and resolved URLs: they die
        # with the conversion and never reach the process-level font cache.
        self._rules = parse_font_face_rules(tree)
        self._available = _load_backend()
        # Per-conversion result of each rule URL: a :class:`LoadedFont`, or
        # ``False`` when the font could not be loaded. A URL is therefore
        # fetched at most once even when several text nodes use the font.
        self._rule_results = {}

    def get_font_face(self, node, family, slant, weight):
        """Return a cairo font face for ``family``, or :data:`None`.

        ``node`` is the text node being rendered: its URL fetcher provides the
        font bytes, so the conversion security policy applies.

        :data:`None` means that no ``@font-face`` rule matches and the caller
        must fall back to the system fonts, preserving the historical
        behavior.

        """
        rule = self._match_rule(family, slant, weight)
        if rule is None or not self._available:
            return None
        loaded = self._get_loaded_font(node, rule)
        if loaded is None:
            # The font could not be fetched or loaded: the caller falls back
            # to the system fonts, preserving the historical behavior.
            return None
        return loaded.font_face

    def _match_rule(self, family, slant, weight):
        """Find the best rule for a family and a style/weight request."""
        candidates = [rule for rule in self._rules if rule.family == family]
        if not candidates:
            return None
        exact = [
            rule for rule in candidates
            if rule.slant == slant and rule.weight == weight]
        if exact:
            return exact[0]
        # Prefer rules matching the requested style, then the weight, and
        # finally give up on style matching rather than failing the text.
        matching_style = [rule for rule in candidates if rule.slant == slant]
        if matching_style:
            candidates = matching_style
        matching_weight = [
            rule for rule in candidates if rule.weight == weight]
        if matching_weight:
            return matching_weight[0]
        return candidates[0]

    def _get_loaded_font(self, node, rule):
        """Fetch (or reuse from the cache) the font of a rule."""
        rule_key = rule.url.geturl()
        if rule_key in self._rule_results:
            return self._rule_results[rule_key] or None
        # The URL fetcher enforces the caller security policy: with the
        # default safe fetcher, non data-URLs are never read.
        try:
            font_bytes = node.fetch_url(rule.url, 'font/ttf')
        except Exception as exception:
            logger.warning(
                'Could not fetch font "%s": %s; using system fonts instead',
                rule_key, exception)
            self._rule_results[rule_key] = False
            return None
        cache_key = hashlib.sha256(font_bytes).digest()
        loaded = _loaded_fonts.get(cache_key)
        if loaded is None:
            loaded = self._load_font(font_bytes, rule)
            if loaded is None:
                # The font could not be loaded. The failure is remembered for
                # this conversion only, so another conversion gets to retry.
                self._rule_results[rule_key] = False
                return None
            _loaded_fonts[cache_key] = loaded
        self._rule_results[rule_key] = loaded
        return loaded

    def _load_font(self, font_bytes, rule):
        """Build a cairo FreeType face from TTF/OTF bytes."""
        font_size = len(font_bytes)
        # create_string_buffer adds a trailing NUL: pass the original size to
        # FreeType, the extra byte is never part of the font data. The buffer
        # is owned by the returned LoadedFont for the whole process lifetime.
        font_buffer = ctypes.create_string_buffer(font_bytes)
        ft_face = ctypes.c_void_p()
        error = _ft_library.FT_New_Memory_Face(
            _ft_library_ref,
            ctypes.cast(font_buffer, ctypes.POINTER(ctypes.c_ubyte)),
            font_size, 0, ctypes.byref(ft_face))
        if error:
            logger.warning(
                'Could not load font "%s" (FreeType error %s); using system '
                'fonts instead', rule.url.geturl(), error)
            return None

        pointer = _cairo_create_font_face(ft_face, 0)
        if not pointer or _cairo_font_face_status(pointer):
            logger.warning(
                'Could not register font "%s"; using system fonts instead',
                rule.url.geturl())
            _ft_library.FT_Done_Face(ft_face)
            return None

        # cairocffi only binds the toy font face, so wrap the raw cairo-ft
        # pointer in its base FontFace wrapper. The wrapper owns a reference
        # on the cairo font face and destroys it on garbage collection.
        from cairocffi import ffi
        from cairocffi.fonts import FontFace
        font_face = FontFace(ffi.cast('cairo_font_face_t *', pointer))
        return LoadedFont(font_buffer, font_size, ft_face, font_face)

    def cleanup(self):
        """End the conversion.

        The document rules are simply dropped. Font resources stay in the
        process-level cache because the cairo FreeType backend may still read
        the font memory through its scaled-font cache; freeing it here would
        corrupt later conversions.

        """
        self._rules = []
        self._rule_results = {}
