"""Supabase JWT verification for direct browser-to-service integration."""

from typing import Any

import jwt
from fastapi import Depends, Header, HTTPException, status
from jwt import PyJWKClient

from .config import settings


_jwks_client: PyJWKClient | None = None


def _unauthorized(message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"code": "INVALID_AUTH_TOKEN", "message": message},
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """Validate a Supabase access token and return only safe identity claims."""
    if not authorization:
        raise _unauthorized("Authorization Bearer token is required")
    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise _unauthorized("Authorization header must use Bearer token format")
    global _jwks_client
    try:
        if _jwks_client is None:
            _jwks_client = PyJWKClient(settings.supabase_jwks_url, cache_keys=True, cache_jwk_set=True, lifespan=600)
        signing_key = _jwks_client.get_signing_key_from_jwt(parts[1])
        claims = jwt.decode(
            parts[1],
            signing_key.key,
            algorithms=["RS256", "RS384", "RS512", "ES256", "ES384", "ES512", "EdDSA"],
            audience=settings.supabase_jwt_audience,
            issuer=settings.supabase_issuer,
            options={"require": ["sub", "iss", "aud", "exp"]},
        )
        return {"sub": str(claims["sub"]), "role": claims.get("role")}
    except Exception as error:
        # Do not leak JWKS, token, or provider details to the browser.
        raise _unauthorized("Invalid or expired access token") from error
