__all__ = ["OlmoError", "OlmoConfigurationError", "OlmoCliError", "OlmoNetworkError"]


class OlmoError(Exception):
    """
    Base class for all custom OLMo exceptions.
    """


class OlmoConfigurationError(OlmoError):
    """
    An error with a configuration file.
    """


class OlmoCliError(OlmoError):
    """
    An error from incorrect CLI usage.
    """


class OlmoNetworkError(OlmoError):
    """
    An error with a network request.
    """
