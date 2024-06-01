#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from concurrent import futures
import argparse
import logging
import time

import grpc
import concurrent_shard_pb2
import concurrent_shard_pb2_grpc


class ConcurrentShardServicer(concurrent_shard_pb2_grpc.ConcurrentShardServicer):

    def Start(self, request: concurrent_shard_pb2.TestIns, context):
        logging.info(f"received request to sleep {request.secs}")
        time.sleep(request.secs)
        return concurrent_shard_pb2.Empty()


def serve(port: int):
    logging.basicConfig(level=logging.INFO)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
    )
    concurrent_shard_pb2_grpc.add_ConcurrentShardServicer_to_server(
        ConcurrentShardServicer(), server
    )
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    server.start()
    logging.info(f"server started, listening on {listen_addr}")
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    serve(args.port)
