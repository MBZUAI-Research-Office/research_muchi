# DBRX distributed inference with weights batched by expert
## quickstart
workdir:
```
cd batch2/
```
launch all shards:
```
python launch_moe_shards_batch2.py --model-path ~/dbrx-base/distributable/batch2/
```
terminate all shards:
```
python launch_moe_shards_batch2.py --model-path ~/dbrx-base/distributable/batch2/ --terminate
```
