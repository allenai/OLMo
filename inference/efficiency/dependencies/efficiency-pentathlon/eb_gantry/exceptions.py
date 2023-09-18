class GantryError(Exception):
    """
    Base exception for all error types that Gantry might raise.
    """


class GitError(GantryError):
    pass


class DirtyRepoError(GitError):
    pass


class InvalidRemoteError(GitError):
    pass


class ConfigurationError(GantryError):
    pass


class ExperimentFailedError(GantryError):
    pass


class EntrypointChecksumError(GantryError):
    pass


class GitHubTokenSecretNotFound(GantryError):
    pass


class TermInterrupt(GantryError):
    pass
