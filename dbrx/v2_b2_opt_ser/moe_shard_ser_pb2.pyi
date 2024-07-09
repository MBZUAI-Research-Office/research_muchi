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
    __slots__ = ("data", "exec_time")
    DATA_FIELD_NUMBER: _ClassVar[int]
    EXEC_TIME_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    exec_time: float
    def __init__(self, data: _Optional[bytes] = ..., exec_time: _Optional[float] = ...) -> None: ...
