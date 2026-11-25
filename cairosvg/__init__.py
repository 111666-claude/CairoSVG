"""
CairoSVG - A simple SVG converter based on Cairo.

"""

import sys
from pathlib import Path

if hasattr(sys, 'frozen'):
    if hasattr(sys, '_MEIPASS'):
        # Frozen with PyInstaller
        # See https://github.com/Kozea/WeasyPrint/pull/540
        ROOT = Path(sys._MEIPASS) / 'cairosvg'
    else:
        # Frozen with something else (py2exe, etc.)
        # See https://github.com/Kozea/WeasyPrint/pull/269
        ROOT = Path(sys.executable).parent
else:
    ROOT = Path(__file__).resolve().parent

VERSION = __version__ = (ROOT / 'VERSION').read_text().strip()


# VERSION is used in the "url" module imported by "surface"
from . import surface  # noqa isort:skip
from .css import (  # noqa isort:skip
    CSSError, CSSParseError, CSSParserError, CSSSyntaxError, InvalidCSS,
    InvalidStylesheet, ParseError, StylesheetError, StylesheetParseError,
    StylesheetSyntaxError)


SURFACES = {
    'PDF': surface.PDFSurface,
    'PNG': surface.PNGSurface,
    'PS': surface.PSSurface,
    'EPS': surface.EPSSurface,
    'SVG': surface.SVGSurface,
}


def svg2svg(bytestring=None, stylesheet=None, *, file_obj=None, url=None,
            dpi=96, parent_width=None, parent_height=None, scale=1,
            unsafe=False, background_color=None, negate_colors=False,
            invert_images=False, write_to=None, output_width=None,
            output_height=None, **kwargs):
    return surface.SVGSurface.convert(
        bytestring=bytestring, stylesheet=stylesheet, file_obj=file_obj,
        url=url, dpi=dpi, parent_width=parent_width,
        parent_height=parent_height, scale=scale,
        background_color=background_color,
        negate_colors=negate_colors, invert_images=invert_images,
        unsafe=unsafe, write_to=write_to, output_width=output_width,
        output_height=output_height, **kwargs)


def svg2png(bytestring=None, stylesheet=None, *, file_obj=None, url=None,
            dpi=96, parent_width=None, parent_height=None, scale=1,
            unsafe=False, background_color=None, negate_colors=False,
            invert_images=False, write_to=None, output_width=None,
            output_height=None, **kwargs):
    return surface.PNGSurface.convert(
        bytestring=bytestring, stylesheet=stylesheet, file_obj=file_obj,
        url=url, dpi=dpi, parent_width=parent_width,
        parent_height=parent_height, scale=scale,
        background_color=background_color, negate_colors=negate_colors,
        invert_images=invert_images, unsafe=unsafe, write_to=write_to,
        output_width=output_width, output_height=output_height, **kwargs)


def svg2pdf(bytestring=None, stylesheet=None, *, file_obj=None, url=None,
            dpi=96, parent_width=None, parent_height=None, scale=1,
            unsafe=False, background_color=None, negate_colors=False,
            invert_images=False, write_to=None, output_width=None,
            output_height=None, **kwargs):
    return surface.PDFSurface.convert(
        bytestring=bytestring, stylesheet=stylesheet, file_obj=file_obj,
        url=url, dpi=dpi, parent_width=parent_width,
        parent_height=parent_height, scale=scale,
        background_color=background_color, negate_colors=negate_colors,
        invert_images=invert_images, unsafe=unsafe, write_to=write_to,
        output_width=output_width, output_height=output_height, **kwargs)


def svg2ps(bytestring=None, stylesheet=None, *, file_obj=None, url=None,
           dpi=96, parent_width=None, parent_height=None, scale=1,
           unsafe=False, background_color=None, negate_colors=False,
           invert_images=False, write_to=None, output_width=None,
           output_height=None, **kwargs):
    return surface.PSSurface.convert(
        bytestring=bytestring, stylesheet=stylesheet, file_obj=file_obj,
        url=url, dpi=dpi, parent_width=parent_width,
        parent_height=parent_height, scale=scale,
        background_color=background_color, negate_colors=negate_colors,
        invert_images=invert_images, unsafe=unsafe, write_to=write_to,
        output_width=output_width, output_height=output_height, **kwargs)


def svg2eps(bytestring=None, stylesheet=None, *, file_obj=None, url=None,
            dpi=96, parent_width=None, parent_height=None, scale=1,
            unsafe=False, background_color=None, negate_colors=False,
            invert_images=False, write_to=None, output_width=None,
            output_height=None, **kwargs):
    return surface.EPSSurface.convert(
        bytestring=bytestring, stylesheet=stylesheet, file_obj=file_obj,
        url=url, dpi=dpi, parent_width=parent_width,
        parent_height=parent_height, scale=scale,
        background_color=background_color, negate_colors=negate_colors,
        invert_images=invert_images, unsafe=unsafe, write_to=write_to,
        output_width=output_width, output_height=output_height, **kwargs)


if __debug__:
    svg2svg.__doc__ = surface.Surface.convert.__doc__.replace(
        'the format for this class', 'SVG')
    svg2png.__doc__ = surface.Surface.convert.__doc__.replace(
        'the format for this class', 'PNG')
    svg2pdf.__doc__ = surface.Surface.convert.__doc__.replace(
        'the format for this class', 'PDF')
    svg2ps.__doc__ = surface.Surface.convert.__doc__.replace(
        'the format for this class', 'PS')
    svg2eps.__doc__ = surface.Surface.convert.__doc__.replace(
        'the format for this class', 'EPS')
