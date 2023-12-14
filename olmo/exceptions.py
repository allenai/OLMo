__all__ = ["OlmoError", "OlmoConfigurationError", "OlmoCliError", "OlmoEnvironmentError", "OlmoNetworkError"]


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


class OlmoEnvironmentError(OlmoError):
    """
    An error from incorrect environment variables.
    """


class OlmoNetworkError(OlmoError):
    """
    An error with a network request.
    """


class OlmoThreadError(Exception):
    """
    Raised when a thread fails.
    """
