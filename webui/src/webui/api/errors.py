"""Shared API plumbing (errors, response helpers)."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class WebUIError(Exception):
    """API-level error with a JSON-able ``detail`` body."""

    def __init__(self, status_code: int, detail: dict[str, Any],
                 message: Optional[str] = None):
        super().__init__(message or str(detail))
        self.status_code = status_code
        self.detail = detail

    def to_response(self) -> JSONResponse:
        return JSONResponse(status_code=self.status_code,
                            content={"error": self.detail})


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(WebUIError)
    async def _webui_error(_: Request, exc: WebUIError):
        return exc.to_response()

    @app.exception_handler(KeyError)
    async def _key_error(_: Request, exc: KeyError):
        return JSONResponse(status_code=404,
                            content={"error": {"message": str(exc.args[0])}})

    @app.exception_handler(ValueError)
    async def _value_error(_: Request, exc: ValueError):
        return JSONResponse(status_code=400,
                            content={"error": {"message": str(exc.args[0])}})


__all__ = ["WebUIError", "register_error_handlers"]
