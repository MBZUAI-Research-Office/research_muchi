from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class TestIns(_message.Message):
    __slots__ = ("batch_size",)
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    batch_size: int
    def __init__(self, batch_size: _Optional[int] = ...) -> None: ...

class ShardOuts(_message.Message):
    __slots__ = ("data", "metadata")
    DATA_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    metadata: bytes
    def __init__(self, data: _Optional[bytes] = ..., metadata: _Optional[bytes] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
