__all__ = ["DolmaError", "DolmaConfigurationError"]


class DolmaError(Exception):
    """
    Base class for all custom DOLMA exceptions.
    """


class DolmaConfigurationError(DolmaError):
    """
    An error with a configuration file.
    """
