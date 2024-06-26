# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import shard_pb2 as shard__pb2


class ShardStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Receive = channel.unary_unary(
                '/Shard/Receive',
                request_serializer=shard__pb2.ShardOuts.SerializeToString,
                response_deserializer=shard__pb2.Empty.FromString,
                )
        self.StartTest = channel.unary_unary(
                '/Shard/StartTest',
                request_serializer=shard__pb2.Inputs.SerializeToString,
                response_deserializer=shard__pb2.Empty.FromString,
                )


class ShardServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Receive(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StartTest(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ShardServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Receive': grpc.unary_unary_rpc_method_handler(
                    servicer.Receive,
                    request_deserializer=shard__pb2.ShardOuts.FromString,
                    response_serializer=shard__pb2.Empty.SerializeToString,
            ),
            'StartTest': grpc.unary_unary_rpc_method_handler(
                    servicer.StartTest,
                    request_deserializer=shard__pb2.Inputs.FromString,
                    response_serializer=shard__pb2.Empty.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Shard', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Shard(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Receive(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Shard/Receive',
            shard__pb2.ShardOuts.SerializeToString,
            shard__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def StartTest(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Shard/StartTest',
            shard__pb2.Inputs.SerializeToString,
            shard__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
