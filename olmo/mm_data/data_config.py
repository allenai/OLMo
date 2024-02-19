import hashlib
from dataclasses import dataclass, field
from os.path import join

import numpy as np


# TODO need these constant somewhere, is the best place?
# Its pretty annoying to have to pass this object around


@dataclass
class MMStorageConfig:
  document_end_token: int = 0
  image_start_token_id: int = 50258
  mask_start_token_id: int = 50259
  mask_end_token_id: int = 50260
  object_id_length: int = 32
  object_id_hash: str = "sha256"
  image_start_bytes: bytes = field(init=False)
  mask_start_bytes: bytes = field(init=False)
  mask_end_bytes: bytes = field(init=False)
  doc_end_bytes: bytes = field(init=False)

  def __post_init__(self):
    self.image_start_bytes = np.array(self.image_start_token_id, np.uint16).tobytes()
    self.mask_start_bytes = np.array(self.mask_start_token_id, np.uint16).tobytes()
    self.mask_end_bytes = np.array(self.mask_end_token_id, np.uint16).tobytes()
    self.doc_end_bytes = np.array(self.document_end_token, np.uint16).tobytes()

  def get_object_id(self, data: bytes) -> bytes:
    if self.object_id_hash == "sha256":
      return hashlib.sha256(data).digest()
    else:
      raise NotImplementedError(self.object_id_hash)

