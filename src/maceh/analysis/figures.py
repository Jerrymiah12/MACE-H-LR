"""Deterministic figure output.

Every figure this project writes goes through :func:`save_figure`. Matplotlib
stamps wall-clock time into PDF (``/CreationDate``), PostScript and SVG output,
so two runs of the same command produce different bytes and any checksum
recorded against them is unverifiable by construction.

This module *omits* those keys rather than freezing them to a constant epoch.
Nothing then has to be kept in sync, and there is no baked-in date that later
reads as a false claim about when a figure was made.

What is stripped, and what determinism that actually buys:

===========  =================================================================
``.pdf``     ``/CreationDate`` and ``/Producer`` (the latter carries the
             matplotlib version, so keeping it makes output version-dependent)
``.ps``      ``/CreationDate``
``.eps``     as ``.ps``
``.svg``     ``Date``, plus a fixed ``svg.hashsalt`` -- without the salt,
             element ids are randomised per run and the file still differs
``.png``     already deterministic; ``Software`` is dropped for consistency
===========  =================================================================

**Scope.** Byte-identical output is a promise about one pinned environment, not
about these formats in general: font subsetting depends on the FreeType build,
and rasterisation can differ across platforms. Expect checksums to agree on
re-run on one machine and to disagree across machines. ``results/README.md``
states what the artifact manifests therefore claim.
"""

import os
from pathlib import Path

import matplotlib

# Belt and braces. Any figure that escapes this module -- third-party code, or a
# call site added later that forgets to route through save_figure -- still gets a
# fixed timestamp instead of the wall clock, because matplotlib honours
# SOURCE_DATE_EPOCH. setdefault, so an explicit value from the caller wins.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

#: Fixed salt for SVG element ids; any constant works, it just must not vary.
SVG_HASHSALT = "maceh"

#: Metadata keys that carry a timestamp or a toolchain version, by file suffix.
_NONDETERMINISTIC_KEYS = {
    ".pdf": ("CreationDate", "Producer"),
    ".ps": ("CreationDate",),
    ".eps": ("CreationDate",),
    ".svg": ("Date",),
    ".png": ("Software",),
}


def save_figure(fig, path, **kwargs):
    """Write ``fig`` to ``path`` with non-deterministic metadata removed.

    Accepts and forwards every ``Figure.savefig`` keyword. An explicit
    ``metadata`` mapping is honoured; only keys it does not already set are
    filled in, so a caller can still stamp its own ``Title`` or ``Author``.

    Returns the path written, as a :class:`~pathlib.Path`.
    """
    path = Path(path)
    metadata = dict(kwargs.pop("metadata", None) or {})
    for key in _NONDETERMINISTIC_KEYS.get(path.suffix.lower(), ()):
        metadata.setdefault(key, None)
    if metadata:
        # Formats without metadata support reject the keyword outright, so only
        # pass it when this suffix actually has something to strip.
        kwargs["metadata"] = metadata
    with matplotlib.rc_context({"svg.hashsalt": SVG_HASHSALT}):
        fig.savefig(path, **kwargs)
    return path


def save_figure_formats(fig, output_dir, stem, formats=("png", "pdf"),
                        dpi=300, **kwargs):
    """Write one figure as several formats, the way the campaigns want it.

    Replaces the near-identical ``save``/``save_figure`` helpers that each
    figure script used to carry. ``dpi`` applies to raster formats only; vector
    formats get matplotlib's default, matching the previous behaviour.

    Returns the list of paths written.
    """
    raster = {"png", "jpg", "jpeg", "tif", "tiff"}
    return [
        save_figure(fig, Path(output_dir) / f"{stem}.{fmt}",
                    dpi=dpi if fmt.lower() in raster else None, **kwargs)
        for fmt in formats
    ]


def pdf_pages(path, **kwargs):
    """A :class:`~matplotlib.backends.backend_pdf.PdfPages` without a timestamp.

    ``PdfPages`` takes its metadata at construction rather than per page, so
    multi-page PDFs cannot go through :func:`save_figure`.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    metadata = dict(kwargs.pop("metadata", None) or {})
    for key in _NONDETERMINISTIC_KEYS[".pdf"]:
        metadata.setdefault(key, None)
    return PdfPages(path, metadata=metadata, **kwargs)


__all__ = ["SVG_HASHSALT", "save_figure", "save_figure_formats", "pdf_pages"]
