"""Errors raised while parsing or validating a support-gate result."""


class SupportGateError(ValueError):
    """The provider result is not a safe support classification."""
