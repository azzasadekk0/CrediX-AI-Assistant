"""Safe project-level errors for LLM configuration and providers."""


class LLMError(RuntimeError):
    pass


class LLMConfigurationError(LLMError):
    pass


class LLMProviderError(LLMError):
    pass


class LLMAuthenticationError(LLMProviderError):
    pass


class LLMTimeoutError(LLMProviderError):
    pass


class LLMRateLimitError(LLMProviderError):
    pass


class LLMResponseError(LLMProviderError):
    pass
