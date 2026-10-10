"""Generation-layer errors; no provider credentials are ever included."""


class GroundedGenerationError(RuntimeError):
    pass


class GeneralGenerationError(RuntimeError):
    """Raised when the standalone, non-grounded generation contract fails."""

    pass


class StructuredOutputError(GroundedGenerationError):
    pass


class CitationValidationError(GroundedGenerationError):
    def __init__(self, invalid_handles: list[str], message: str = "citation validation failed") -> None:
        super().__init__(message)
        self.invalid_handles = invalid_handles
