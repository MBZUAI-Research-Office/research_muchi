# DBRX distributed inference with weights batched by expert
## quickstart
workdir:
```
cd b2_opt_ser/
```
launch all shards:
```
python launch_moe_shards_ser.py --model-path ~/dbrx-base/distributable/batch2/
```
terminate all shards:
```
python launch_moe_shards_ser.py --model-path ~/dbrx-base/distributable/batch2/ --terminate
```
