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




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x12local_driver.proto\"/\n\tUsrInputs\x12\x0e\n\x06prompt\x18\x01 \x01(\t\x12\x12\n\nmax_tokens\x18\x02 \x01(\x05\"n\n\nUsrOutputs\x12\x13\n\x0bprompt_time\x18\x01 \x01(\x01\x12\x14\n\x0cprompt_t_cnt\x18\x02 \x01(\x05\x12\x10\n\x08gen_time\x18\x03 \x01(\x01\x12\x11\n\tgen_t_cnt\x18\x04 \x01(\x05\x12\x10\n\x08response\x18\x05 \x01(\t\"M\n\x0cMoeShardOuts\x12\x0b\n\x03url\x18\x01 \x01(\t\x12\x11\n\tlayer_num\x18\x02 \x01(\x05\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\x0c\x12\x0f\n\x07\x61rr_map\x18\x04 \x01(\x0c\"\x07\n\x05\x45mpty2U\n\x0bLocalDriver\x12\"\n\x05Start\x12\n.UsrInputs\x1a\x0b.UsrOutputs\"\x00\x12\"\n\x07Receive\x12\r.MoeShardOuts\x1a\x06.Empty\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'local_driver_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_USRINPUTS']._serialized_start=22
  _globals['_USRINPUTS']._serialized_end=69
  _globals['_USROUTPUTS']._serialized_start=71
  _globals['_USROUTPUTS']._serialized_end=181
  _globals['_MOESHARDOUTS']._serialized_start=183
  _globals['_MOESHARDOUTS']._serialized_end=260
  _globals['_EMPTY']._serialized_start=262
  _globals['_EMPTY']._serialized_end=269
  _globals['_LOCALDRIVER']._serialized_start=271
  _globals['_LOCALDRIVER']._serialized_end=356
# @@protoc_insertion_point(module_scope)
