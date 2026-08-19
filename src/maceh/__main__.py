"""Entry point for ``python -m maceh``.

Delegates to the same :func:`maceh.cli.main` the ``maceh`` console script in
``pyproject.toml`` points at, so both dispatch paths share one implementation.
"""
from .cli import main

raise SystemExit(main())
