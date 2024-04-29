from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Inputs(_message.Message):
    __slots__ = ("activated_experts", "data")
    ACTIVATED_EXPERTS_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    activated_experts: bytes
    data: bytes
    def __init__(self, activated_experts: _Optional[bytes] = ..., data: _Optional[bytes] = ...) -> None: ...

class Outputs(_message.Message):
    __slots__ = ("data",)
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    def __init__(self, data: _Optional[bytes] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
