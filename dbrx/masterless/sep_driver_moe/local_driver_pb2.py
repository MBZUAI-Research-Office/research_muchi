# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: local_driver.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x12local_driver.proto\",\n\x06Inputs\x12\x0e\n\x06prompt\x18\x01 \x01(\t\x12\x12\n\nmax_tokens\x18\x02 \x01(\x05\"k\n\x07Outputs\x12\x13\n\x0bprompt_time\x18\x01 \x01(\x01\x12\x14\n\x0cprompt_t_cnt\x18\x02 \x01(\x05\x12\x10\n\x08gen_time\x18\x03 \x01(\x01\x12\x11\n\tgen_t_cnt\x18\x04 \x01(\x05\x12\x10\n\x08response\x18\x05 \x01(\t\"M\n\x0cMoeShardOuts\x12\x0b\n\x03url\x18\x01 \x01(\t\x12\x11\n\tlayer_num\x18\x02 \x01(\x05\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\x0c\x12\x0f\n\x07\x61rr_map\x18\x04 \x01(\x0c\"\x07\n\x05\x45mpty2O\n\x0bLocalDriver\x12\x1c\n\x05Start\x12\x07.Inputs\x1a\x08.Outputs\"\x00\x12\"\n\x07Receive\x12\r.MoeShardOuts\x1a\x06.Empty\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'local_driver_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_INPUTS']._serialized_start=22
  _globals['_INPUTS']._serialized_end=66
  _globals['_OUTPUTS']._serialized_start=68
  _globals['_OUTPUTS']._serialized_end=175
  _globals['_MOESHARDOUTS']._serialized_start=177
  _globals['_MOESHARDOUTS']._serialized_end=254
  _globals['_EMPTY']._serialized_start=256
  _globals['_EMPTY']._serialized_end=263
  _globals['_LOCALDRIVER']._serialized_start=265
  _globals['_LOCALDRIVER']._serialized_end=344
# @@protoc_insertion_point(module_scope)
