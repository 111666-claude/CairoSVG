"""
Handle CSS stylesheets.

"""

import re

import cssselect2
import tinycss2

from .url import URLFetchingError, parse_url, safe_fetch

# Processing instructions are not part of the ElementTree document built by
# defusedxml (they are dropped while parsing), so xml-stylesheet PIs are found
# by scanning the raw XML source. See https://www.w3.org/TR/xml-stylesheet/.
STYLESHEET_PI = re.compile(r'<\?xml-stylesheet\s+(.*?)\?>', re.DOTALL)
PSEUDO_ATTRIBUTE = re.compile(
    r'([:A-Za-z_][-\w:.]*)\s*=\s*(["\'])(.*?)\2', re.DOTALL)


def parse_xml_stylesheets(tree, url, bytestring):
    """Find the external CSS stylesheets declared by xml-stylesheet PIs.

    Relative ``href`` attributes are resolved against the URL of the SVG
    document. Stylesheets are fetched through the document's URL fetcher, so
    that the current security policy applies: in the default safe mode only
    data URLs can be fetched, any other external stylesheet raises a
    :class:`URLFetchingError`, and a missing or unreadable stylesheet also
    raises a :class:`URLFetchingError` rather than being silently ignored.

    See https://www.w3.org/TR/xml-stylesheet/.

    """
    if bytestring is None:
        return
    # Pseudo-attributes are plain ASCII; an unsupported encoding would have
    # already raised while parsing the XML itself.
    source = bytestring.decode('utf-8', errors='replace')
    seen_urls = set()
    # Alternate stylesheets (title set, not the first preferred one) are
    # ignored: CairoSVG has no mechanism to select an alternate stylesheet.
    preferred_title = None
    for match in STYLESHEET_PI.finditer(source):
        attributes = {
            pseudo_match.group(1): pseudo_match.group(3)
            for pseudo_match in PSEUDO_ATTRIBUTE.finditer(match.group(1))}
        href = attributes.get('href')
        if not href:
            # An xml-stylesheet PI without href is an error, ignore it.
            continue
        if attributes.get('type', 'text/css') != 'text/css':
            continue
        # TODO: support media types
        title = attributes.get('title')
        if title is not None:
            if preferred_title is None:
                preferred_title = title
            elif title != preferred_title:
                # Only the first preferred stylesheet is applied, every
                # other titled stylesheet is an alternate one.
                continue
        css_url = parse_url(href, url)
        css_url_text = css_url.geturl()
        if css_url_text in seen_urls:
            # Duplicated PIs pointing at the same resource are fetched once.
            continue
        seen_urls.add(css_url_text)
        if tree.url_fetcher is safe_fetch and not (
                css_url.scheme == 'data'):
            raise URLFetchingError(
                f'External stylesheet {css_url_text!r} cannot be fetched in '
                'safe mode. Use unsafe=True or provide a custom '
                'url_fetcher to allow external resources.')
        try:
            css_bytes = tree.fetch_url(css_url, 'text/css')
        except URLFetchingError:
            raise
        except Exception as exception:
            raise URLFetchingError(
                f'Could not fetch stylesheet {css_url_text!r}: {exception}'
            ) from exception
        yield tinycss2.parse_stylesheet(
            css_bytes.decode('utf-8', errors='replace'),
            skip_comments=True, skip_whitespace=True)


def find_stylesheets(tree, url, bytestring):
    """Find the stylesheets included in ``tree``."""
    # External stylesheets referenced by xml-stylesheet processing
    # instructions come first, as PIs are declared in the document prolog.
    yield from parse_xml_stylesheets(tree, url, bytestring)
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


def parse_stylesheets(tree, url, bytestring=None):
    """Find and parse the stylesheets in ``tree``.

    Return two :class:`cssselect2.Matcher` objects,
    for normal and !important declarations.

    """
    normal_matcher = cssselect2.Matcher()
    important_matcher = cssselect2.Matcher()
    for stylesheet in find_stylesheets(tree, url, bytestring):
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
    return normal_matcher, important_matcher


def get_declarations(rule):
    """Get the declarations in ``rule``."""
    if rule.type == 'qualified-rule':
        for declaration in tinycss2.parse_declaration_list(
                rule.content, skip_comments=True, skip_whitespace=True):
            value = ''.join(part.serialize() for part in declaration.value)
            # TODO: filter out invalid values
            yield declaration.lower_name, value, declaration.important
