# shard_outs = {}
# compute_fut = executor.submit(
#     self.call_shard_n_all_dispatch,
#     x,
#     jobs,
#     ws,
#     raw_weights.ne_warmup,
#     send_conn,
# )
# comm_fut = executor.submit(self.all_combine, batch_size, resv_conn)
# fut_map = {compute_fut: "moe", comm_fut: "comm"}
# for fut in concurrent.futures.as_completed(fut_map):
#     if fut_map[fut] == "moe":
#         shard_outs.update(fut.result()[0])
#         LOGS["moe_lat"].append(fut.result()[1])
#     else:
#         shard_outs.update(fut.result())

# LOGS["comm_lat"].append(time.perf_counter_ns() - tic - LOGS["moe_lat"][-1])