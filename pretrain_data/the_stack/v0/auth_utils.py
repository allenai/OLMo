import os
from typing import Optional

import fsspec


def fs_auth(token: Optional[str] = None):
    """
    Authenticate with huggingface token to allow dataset access.
    """
    if not token:
        try:
            token = os.environ["HF_TOKEN"]
        except KeyError:
            raise KeyError("Please specify HF_TOKEN in the environment.")
    headers = {"Authorization": f"Bearer {token}"}
    fs = fsspec.filesystem("https", headers=headers)
    return fs
