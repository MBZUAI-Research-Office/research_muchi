from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Inputs(_message.Message):
    __slots__ = ("block_num", "activated_experts", "data")
    BLOCK_NUM_FIELD_NUMBER: _ClassVar[int]
    ACTIVATED_EXPERTS_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    block_num: int
    activated_experts: bytes
    data: bytes
    def __init__(self, block_num: _Optional[int] = ..., activated_experts: _Optional[bytes] = ..., data: _Optional[bytes] = ...) -> None: ...

class Outputs(_message.Message):
    __slots__ = ("data", "exec_time")
    DATA_FIELD_NUMBER: _ClassVar[int]
    EXEC_TIME_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    exec_time: float
    def __init__(self, data: _Optional[bytes] = ..., exec_time: _Optional[float] = ...) -> None: ...
