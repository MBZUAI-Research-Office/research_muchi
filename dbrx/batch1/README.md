# DBRX distributed inference with weights batched by expert
## quickstart
workdir:
```
cd batch1/
```
launch all shards:
```
python launch_moe_shards_batch1.py --model-path ~/dbrx-base/distributable/batch1/
```
terminate all shards:
```
python launch_moe_shards_batch1.py --model-path ~/dbrx-base/distributable/batch1/ --terminate
```
