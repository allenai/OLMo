
class ImageTokenSizer:
  """Determines the number of tokens per an image"""

  def __call__(self, width: int, height: int) -> int:
    """Number of tokens for an image of width x height"""
    raise NotImplementedError()

  def get_id(self) -> str:
    """Persistent ID that can be used to hash this sizing algorithm mechanism"""
    raise NotImplementedError()


class FixedNumberOfToken(ImageTokenSizer):
  def __init__(self, tokens: int):
    self.tokens = tokens

  def __call__(self, width: int, height: int) -> int:
    return self.tokens

  def get_id(self) -> str:
    return f"fixed{self.tokens}"


class PerPatch(ImageTokenSizer):
  def __init__(self, patch_h, patch_w):
    self.patch_h = patch_h
    self.patch_w = patch_w

  def __call__(self, width: int, height: int) -> int:
    n_w = (width + self.patch_w - 1) // self.patch_w
    n_h = (height + self.patch_h - 1) // self.patch_h
    return n_w*n_h

  def get_id(self) -> str:
    return f"per-patch{self.patch_h}x{self.patch_w}"


class Patchify(ImageTokenSizer):
  def __init__(self, patch_w, patch_h, tokens_per_patch):
    self.patch_w = patch_w
    self.patch_h = patch_h
    self.tokens_per_patch = tokens_per_patch

  def __call__(self, width: int, height: int) -> int:
    n_w = (width + self.patch_w - 1) // self.patch_w
    n_h = (height + self.patch_h - 1) // self.patch_h
    # assume one "new-row" token per a row
    return (n_w*n_h)*self.tokens_per_patch + (self.patch_h - 1)

  def get_id(self) -> str:
    return f"patchify{self.patch_w}-{self.patch_h}-{self.tokens_per_patch}"
