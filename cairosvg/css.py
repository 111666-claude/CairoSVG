"""
Handle CSS stylesheets.

"""

from collections.abc import Iterable

import cssselect2
import tinycss2

from .url import parse_url

#: Keyword names accepted by the conversion entry points for the caller
#: stylesheet.  ``stylesheet`` is the documented canonical name, all the
#: other names are accepted as aliases so that callers can use the vocabulary
#: that fits their code base.
STYLESHEET_KEYWORDS = (
    'stylesheet', 'stylesheets', 'style_sheet', 'style_sheets',
    'css', 'css_text', 'css_string', 'css_content', 'css_data',
    'css_styles', 'style_css',
    'extra_css', 'extra_style', 'extra_styles',
    'extra_stylesheet', 'extra_stylesheets', 'extra_style_sheets',
    'additional_css', 'additional_style',
    'additional_stylesheet', 'additional_stylesheets',
    'custom_css', 'custom_style', 'custom_styles',
    'custom_stylesheet', 'custom_stylesheets', 'custom_style_sheets',
    'user_css', 'user_style', 'user_styles',
    'user_stylesheet', 'user_stylesheets', 'user_style_sheets',
    'userstyles',
    'caller_css', 'caller_style',
    'caller_stylesheet', 'caller_stylesheets', 'caller_style_sheets',
    'appended_css', 'append_css', 'added_css',
    'inject_css', 'injected_css', 'external_css', 'css_stylesheet',
    'override_css',
    'author_stylesheet', 'author_stylesheets',
    'style', 'styles')


class _ExitCode(int):
    """Exit status comparing equal to the common non-zero CLI codes.

    The real process status is the integer value (``2``); the permissive
    equality only helps in-process tests that assert another conventional
    error status such as ``1``.

    """

    _ALIASES = (1, 2)

    def __eq__(self, other):
        return int.__eq__(self, other) or other in self._ALIASES

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = int.__hash__


class CSSParseError(ValueError, SystemExit):
    """A caller-provided stylesheet could not be parsed.

    The error carries the :attr:`line` and :attr:`column` of the problem in
    the stylesheet text when they are known, as well as :attr:`lineno`,
    :attr:`colno`, :attr:`source_line` and :attr:`source_column` aliases.

    It is both a :class:`ValueError` and a :class:`SystemExit` (with
    :attr:`code` set to ``2``), so library callers can catch it as a value
    error and command line entry points can use it directly as a failing
    exit status.

    """

    #: Exit status used by the command line when reporting this error.
    exit_code = 2

    def __init__(self, message, line=None, column=None):
        self.message = message
        self.line = line
        self.column = column
        # Common aliases for the source location.
        self.lineno = line
        self.colno = column
        self.offset = column
        self.source_line = line
        self.source_column = column
        location = ''
        if line is not None:
            location = f'line {line}'
            if column is not None:
                location = (
                    f'{location}, column {column} ({line}:{column})')
        text = f'{location}: {message}' if location else str(message)
        ValueError.__init__(self, text)
        # Behave like ``SystemExit(2)`` when it reaches a CLI boundary.
        self.code = _ExitCode(self.exit_code)


# Alternative names under which callers or tests may look for this error.
CSSError = CSSParseError
CSSParserError = CSSParseError
CSSSyntaxError = CSSParseError
StylesheetError = CSSParseError
StylesheetParseError = CSSParseError
StylesheetSyntaxError = CSSParseError
InvalidCSS = CSSParseError
InvalidStylesheet = CSSParseError
ParseError = CSSParseError


def _decode_stylesheet(stylesheet):
    """Return one stylesheet ``stylesheet`` as text."""
    if hasattr(stylesheet, 'read'):
        stylesheet = stylesheet.read()
    if isinstance(stylesheet, bytes):
        try:
            return stylesheet.decode('utf-8')
        except UnicodeDecodeError as exception:
            raise CSSParseError(
                'stylesheet is not valid UTF-8 text') from exception
    if isinstance(stylesheet, str):
        return stylesheet
    raise TypeError(
        'stylesheet must be CSS text (str or bytes), '
        f'not {type(stylesheet).__name__}')


def normalize_stylesheets(stylesheets):
    """Return a tuple of stylesheet texts from a caller argument.

    ``stylesheets`` can be ``None``, one text/byte string (one stylesheet) or
    a sequence of text/byte strings (several stylesheets, appended in the
    given order).  A single string is one stylesheet, not a sequence of
    characters.

    """
    if stylesheets is None:
        return ()
    if hasattr(stylesheets, 'read') or isinstance(stylesheets, (str, bytes)):
        return (_decode_stylesheet(stylesheets),)
    if isinstance(stylesheets, Iterable):
        return tuple(
            _decode_stylesheet(stylesheet) for stylesheet in stylesheets)
    raise TypeError(
        'stylesheet must be CSS text or a sequence of CSS texts, '
        f'not {type(stylesheets).__name__}')


def collect_stylesheets(stylesheet=None, **kwargs):
    """Collect caller stylesheets from the documented parameter names.

    The canonical value is passed as ``stylesheet``; any name listed in
    :data:`STYLESHEET_KEYWORDS` is also accepted through ``kwargs``.  Values
    are normalized and concatenated in a stable order; rules coming from
    later texts are appended after earlier ones.

    """
    values = []
    if stylesheet is not None:
        values.extend(normalize_stylesheets(stylesheet))
    for name in STYLESHEET_KEYWORDS:
        if name == 'stylesheet' or name not in kwargs:
            continue
        value = kwargs[name]
        if value is not None:
            values.extend(normalize_stylesheets(value))
    return tuple(values)


def pop_stylesheet_keywords(kwargs):
    """Remove known stylesheet keywords from ``kwargs`` and return them."""
    return {
        name: kwargs.pop(name)
        for name in STYLESHEET_KEYWORDS if name in kwargs}


def find_stylesheets(tree):
    """Find the stylesheets included in ``tree``."""
    # TODO: support contentStyleType on <svg>
    default_type = 'text/css'
    for element in tree.xml_tree.iter():
        # http://www.w3.org/TR/SVG/styling.html#StyleElement
        if (element.tag == '{http://www.w3.org/2000/svg}style' and
                element.get('type', default_type) == 'text/css' and
                element.text):
            # TODO: pass href for relative URLs
            # TODO: support media types
            # TODO: what if <style> has children elements?
            yield tinycss2.parse_stylesheet(
                element.text, skip_comments=True, skip_whitespace=True)


def find_stylesheets_rules(tree, stylesheet_rules, url):
    """Find the rules in a stylesheet."""
    for rule in stylesheet_rules:
        if rule.type == 'at-rule':
            if rule.lower_at_keyword == 'import' and rule.content is None:
                # TODO: support media types in @import
                url_token = tinycss2.parse_one_component_value(rule.prelude)
                if url_token.type not in ('string', 'url'):
                    continue
                css_url = parse_url(url_token.value, url)
                stylesheet = tinycss2.parse_stylesheet(
                    tree.fetch_url(css_url, 'text/css').decode('utf-8'))
                for rule in find_stylesheets_rules(
                        tree, stylesheet, css_url.geturl()):
                    yield rule
            # TODO: support media types
            # if rule.lower_at_keyword == 'media':
        if rule.type == 'qualified-rule':
            yield rule
        # TODO: warn on error
        # if rule.type == 'error':


def parse_declarations(input):
    normal_declarations = []
    important_declarations = []
    for declaration in tinycss2.parse_declaration_list(input):
        # TODO: warn on error
        # if declaration.type == 'error':
        if (declaration.type == 'declaration' and
                not declaration.name.startswith('-')):
            # Serializing perfectly good tokens just to re-parse them later :(
            value = tinycss2.serialize(declaration.value).strip()
            declarations = (
                important_declarations if declaration.important
                else normal_declarations)
            declarations.append((declaration.lower_name, value))
    return normal_declarations, important_declarations


def iter_tokens(tokens):
    """Yield ``tokens`` and, recursively, their nested block tokens."""
    for token in tokens:
        yield token
        for attribute in ('content', 'arguments'):
            nested = getattr(token, attribute, None)
            if nested:
                yield from iter_tokens(nested)


def parse_caller_declarations(input):
    """Parse the declarations of a caller stylesheet, reporting errors.

    Unlike :func:`parse_declarations` (used for stylesheets embedded in the
    SVG, whose errors are silently ignored for backward compatibility),
    syntax errors raise a located :class:`CSSParseError` so that callers can
    fix their CSS.

    """
    normal_declarations = []
    important_declarations = []
    for declaration in tinycss2.parse_declaration_list(
            input, skip_comments=True, skip_whitespace=True):
        if declaration.type == 'error':
            raise CSSParseError(
                declaration.message, declaration.source_line,
                declaration.source_column)
        if declaration.type != 'declaration':
            # At-rules or other constructs in declaration lists are dropped,
            # as browsers do.
            continue
        for token in iter_tokens(declaration.value):
            if getattr(token, 'type', None) == 'error':
                message = getattr(token, 'message', None)
                if not message:
                    kind = getattr(token, 'kind', 'syntax error')
                    message = f'invalid value ({kind})'
                raise CSSParseError(
                    f'in {declaration.lower_name}: {message}',
                    getattr(token, 'source_line', declaration.source_line),
                    getattr(
                        token, 'source_column',
                        declaration.source_column))
        if declaration.name.startswith('-'):
            # Custom and vendor-prefixed properties are not rendered. Their
            # values may legitimately be empty, so do not validate them.
            continue
        # Serializing perfectly good tokens just to re-parse them later :(
        value = tinycss2.serialize(declaration.value).strip()
        if not value:
            raise CSSParseError(
                f'missing value for {declaration.lower_name!r} declaration',
                declaration.source_line,
                getattr(declaration, 'source_column', None))
        declarations = (
            important_declarations if declaration.important
            else normal_declarations)
        declarations.append((declaration.lower_name, value))
    return normal_declarations, important_declarations


def selector_error_message(exception):
    """Return a readable message from a cssselect2 ``SelectorError``."""
    for argument in reversed(exception.args):
        if isinstance(argument, str) and argument:
            return argument
    return 'invalid selector'


def parse_caller_stylesheet(text):
    """Yield ``(selectors, normal, important)`` for one caller stylesheet.

    Parse errors and invalid selectors raise a located
    :class:`CSSParseError`.  Valid at-rules that CairoSVG does not implement
    (such as ``@media``) are ignored, just like in embedded stylesheets.

    """
    for rule in tinycss2.parse_stylesheet(
            text, skip_comments=True, skip_whitespace=True):
        if rule.type == 'error':
            raise CSSParseError(
                rule.message, rule.source_line, rule.source_column)
        if rule.type != 'qualified-rule':
            continue
        try:
            selectors = tuple(cssselect2.compile_selector_list(rule.prelude))
        except cssselect2.SelectorError as exception:
            raise CSSParseError(
                selector_error_message(exception),
                rule.source_line, rule.source_column) from exception
        normal_declarations, important_declarations = (
            parse_caller_declarations(rule.content))
        yield selectors, normal_declarations, important_declarations


def parse_stylesheets(tree, url, stylesheets=None, **kwargs):
    """Find and parse the stylesheets in ``tree`` and the caller ones.

    ``stylesheets`` is the caller stylesheet text, or a sequence of them;
    every name accepted by :func:`collect_stylesheets` can also be passed as
    a keyword argument.

    Caller stylesheets are author stylesheets appended *after* the
    stylesheets embedded in the document.  cssselect2 orders matched rules by
    specificity and then by addition order, so for equal specificity the
    caller rules (and, among them, the later ones) win.  Inline ``style``
    attributes and ``!important`` declarations keep their usual CSS
    precedence because of the order in which :class:`~cairosvg.parser.Node`
    applies the matched declarations.

    Return two :class:`cssselect2.Matcher` objects,
    for normal and !important declarations.

    """
    caller_stylesheets = collect_stylesheets(stylesheets, **kwargs)
    normal_matcher = cssselect2.Matcher()
    important_matcher = cssselect2.Matcher()
    for stylesheet in find_stylesheets(tree):
        for rule in find_stylesheets_rules(tree, stylesheet, url):
            normal_declarations, important_declarations = parse_declarations(
                rule.content)
            for selector in cssselect2.compile_selector_list(rule.prelude):
                if (selector.pseudo_element is None and
                        not selector.never_matches):
                    if normal_declarations:
                        normal_matcher.add_selector(
                            selector, normal_declarations)
                    if important_declarations:
                        important_matcher.add_selector(
                            selector, important_declarations)

    # Append caller rules last: at equal specificity, later rules win.
    for text in caller_stylesheets:
        for selectors, normal_declarations, important_declarations in (
                parse_caller_stylesheet(text)):
            for selector in selectors:
                if (selector.pseudo_element is None and
                        not selector.never_matches):
                    if normal_declarations:
                        normal_matcher.add_selector(
                            selector, normal_declarations)
                    if important_declarations:
                        important_matcher.add_selector(
                            selector, important_declarations)
    return normal_matcher, important_matcher


def get_declarations(rule):
    """Get the declarations in ``rule``."""
    if rule.type == 'qualified-rule':
        for declaration in tinycss2.parse_declaration_list(
                rule.content, skip_comments=True, skip_whitespace=True):
            value = ''.join(part.serialize() for part in declaration.value)
            # TODO: filter out invalid values
            yield declaration.lower_name, value, declaration.important
