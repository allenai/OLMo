from os.path import join, exists


class ObjectStore:
    """Interface for getting objects from a remote store"""

    # FIXME better to use __in__/__get__?
    def get(self, object_id: bytes) -> bytes:
        raise NotImplementedError()

    def contains(self, object_id: bytes) -> bool:
        raise NotImplementedError()

    def write(self, object_id: bytes, data: bytes):
        raise NotImplementedError()


class FileStore(ObjectStore):
    def __init__(self, src_dir):
        self.src_dir = src_dir

    def _id_to_filename(self, object_id: bytes):
        return join(self.src_dir, object_id.hex() + ".bin")

    def contains(self, object_id: bytes) -> bool:
        return exists(self._id_to_filename(object_id))

    def get(self, object_id):
        with open(self._id_to_filename(object_id), "rb") as f:
            return f.read()

    def write(self, object_id, data):
        with open(self._id_to_filename(object_id), "wb") as f:
            f.write(data)


class InMemoryStore(ObjectStore):

    def __init__(self):
        self.store = {}

    def get(self, object_id: bytes) -> bytes:
        return self.store[object_id]

    def contains(self, object_id: bytes) -> bool:
        return object_id in self.store

    def write(self, object_id: bytes, data: bytes):
        self.store[object_id] = data
