"""A tiny mock auth server for local SSO bootstrap development."""

from __future__ import annotations

import argparse

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse
import uvicorn


def create_mock_auth_app() -> FastAPI:
    """Create a FastAPI app that mimics the OAuth authorize/login flow."""
    app = FastAPI(title="Mock Worker Auth")

    @app.get("/oauth/authorize")
    async def authorize(
        client_id: str = Query(...),
        response_type: str = Query(...),
        redirect_uri: str = Query(...),
    ) -> RedirectResponse:
        if response_type != "code":
            raise HTTPException(status_code=400, detail="unsupported_response_type")
        if not client_id.strip():
            raise HTTPException(status_code=400, detail="missing_client_id")
        separator = "&" if "?" in redirect_uri else "?"
        return RedirectResponse(url=f"{redirect_uri}{separator}code=mock-code", status_code=302)

    @app.get("/user-privilege/login")
    async def login(code: str = Query(...)) -> JSONResponse:
        if code != "mock-code":
            raise HTTPException(status_code=400, detail="invalid_code")
        return JSONResponse(
            content={
                "body": {
                    "userNo": "mock-user-123",
                    "ystId": "mock-yst-123",
                },
                "errorMsg": None,
                "returnCode": "SUC0000",
            },
            headers={"Authorization": "Bearer mock-token"},
        )

    return app


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the mock auth server."""
    parser = argparse.ArgumentParser(description="Mock worker auth server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    return parser.parse_args()


def cli_main() -> None:
    """Run the mock auth server with uvicorn."""
    args = parse_args()
    uvicorn.run(create_mock_auth_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    cli_main()
