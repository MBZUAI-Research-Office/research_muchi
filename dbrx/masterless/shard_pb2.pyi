from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ExpertOuts(_message.Message):
    __slots__ = ("data", "arr_map")
    DATA_FIELD_NUMBER: _ClassVar[int]
    ARR_MAP_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    arr_map: bytes
    def __init__(self, data: _Optional[bytes] = ..., arr_map: _Optional[bytes] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
