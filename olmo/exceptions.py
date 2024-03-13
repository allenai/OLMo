__all__ = [
    "OLMoError",
    "OLMoConfigurationError",
    "OLMoCliError",
    "OLMoEnvironmentError",
    "OLMoNetworkError",
    "OLMoCheckpointError",
]


class OLMoError(Exception):
    """
    Base class for all custom OLMo exceptions.
    """


class OLMoConfigurationError(OLMoError):
    """
    An error with a configuration file.
    """


class OLMoCliError(OLMoError):
    """
    An error from incorrect CLI usage.
    """


class OLMoEnvironmentError(OLMoError):
    """
    An error from incorrect environment variables.
    """


class OLMoNetworkError(OLMoError):
    """
    An error with a network request.
    """


class OLMoCheckpointError(OLMoError):
    """
    An error occurred reading or writing from a checkpoint.
    """


class OLMoThreadError(Exception):
    """
    Raised when a thread fails.
    """
