"""The OpenAI-compatible HTTP interface behind ``yonga serve``.

An interface layer, like :mod:`yonga.cli`: it parses requests, calls
:class:`~yonga.application.serve.ModelServer`, and renders what comes back. It
needs the optional ``serve`` extra (FastAPI and uvicorn), so nothing outside
this package imports it at module scope.
"""
