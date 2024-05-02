from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Inputs(_message.Message):
    __slots__ = ("batch_size", "data")
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    batch_size: int
    data: bytes
    def __init__(self, batch_size: _Optional[int] = ..., data: _Optional[bytes] = ...) -> None: ...

class Outputs(_message.Message):
    __slots__ = ("data",)
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    def __init__(self, data: _Optional[bytes] = ...) -> None: ...
