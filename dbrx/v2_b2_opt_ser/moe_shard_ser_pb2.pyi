from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Inputs(_message.Message):
    __slots__ = ("data",)
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    def __init__(self, data: _Optional[bytes] = ...) -> None: ...

class Outputs(_message.Message):
    __slots__ = ("data", "start", "end")
    DATA_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    start: float
    end: float
    def __init__(self, data: _Optional[bytes] = ..., start: _Optional[float] = ..., end: _Optional[float] = ...) -> None: ...
