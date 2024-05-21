#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

import argparse
import asyncio
import logging

import numpy as np

import grpc
import test_expert_pb2
import test_expert_pb2_grpc


class ExpertServicer(test_expert_pb2_grpc.ExpertServicer):

    async def Execute(
        self, request: test_expert_pb2.Input, context: grpc.aio.ServicerContext
    ):
        arr = np.frombuffer(request.data, dtype=np.float16)
        return test_expert_pb2.Output(data=arr.tobytes())


async def serve(port: int):
    server = grpc.aio.server()
    test_expert_pb2_grpc.add_ExpertServicer_to_server(ExpertServicer(), server)
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    logging.info("Starting server on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(args.port))
