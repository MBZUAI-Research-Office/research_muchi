from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class TestIns(_message.Message):
    __slots__ = ("secs",)
    SECS_FIELD_NUMBER: _ClassVar[int]
    secs: int
    def __init__(self, secs: _Optional[int] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
