from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ShardOuts(_message.Message):
    __slots__ = ("url", "block_num", "data", "arr_map")
    URL_FIELD_NUMBER: _ClassVar[int]
    BLOCK_NUM_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    ARR_MAP_FIELD_NUMBER: _ClassVar[int]
    url: str
    block_num: int
    data: bytes
    arr_map: bytes
    def __init__(self, url: _Optional[str] = ..., block_num: _Optional[int] = ..., data: _Optional[bytes] = ..., arr_map: _Optional[bytes] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
