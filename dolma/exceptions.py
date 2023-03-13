__all__ = ["DolmaError", "DolmaConfigurationError", "DolmaCliError"]


class DolmaError(Exception):
    """
    Base class for all custom DOLMA exceptions.
    """


class DolmaConfigurationError(DolmaError):
    """
    An error with a configuration file.
    """


class DolmaCliError(DolmaError):
    """
    An error from incorrect CLI usage.
    """
