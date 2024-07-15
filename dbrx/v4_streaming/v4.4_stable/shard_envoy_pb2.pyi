from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class UsrIns(_message.Message):
    __slots__ = ("prompt", "max_tokens")
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    prompt: str
    max_tokens: int
    def __init__(self, prompt: _Optional[str] = ..., max_tokens: _Optional[int] = ...) -> None: ...

class UsrOuts(_message.Message):
    __slots__ = ("prompt_time", "prompt_t_cnt", "gen_time", "gen_t_cnt", "response", "avg_moe_lat", "avg_comm_lat", "avg_experts_act")
    PROMPT_TIME_FIELD_NUMBER: _ClassVar[int]
    PROMPT_T_CNT_FIELD_NUMBER: _ClassVar[int]
    GEN_TIME_FIELD_NUMBER: _ClassVar[int]
    GEN_T_CNT_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    AVG_MOE_LAT_FIELD_NUMBER: _ClassVar[int]
    AVG_COMM_LAT_FIELD_NUMBER: _ClassVar[int]
    AVG_EXPERTS_ACT_FIELD_NUMBER: _ClassVar[int]
    prompt_time: float
    prompt_t_cnt: int
    gen_time: float
    gen_t_cnt: int
    response: str
    avg_moe_lat: float
    avg_comm_lat: float
    avg_experts_act: float
    def __init__(self, prompt_time: _Optional[float] = ..., prompt_t_cnt: _Optional[int] = ..., gen_time: _Optional[float] = ..., gen_t_cnt: _Optional[int] = ..., response: _Optional[str] = ..., avg_moe_lat: _Optional[float] = ..., avg_comm_lat: _Optional[float] = ..., avg_experts_act: _Optional[float] = ...) -> None: ...

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

class Identifier(_message.Message):
    __slots__ = ("li",)
    LI_FIELD_NUMBER: _ClassVar[int]
    li: int
    def __init__(self, li: _Optional[int] = ...) -> None: ...
