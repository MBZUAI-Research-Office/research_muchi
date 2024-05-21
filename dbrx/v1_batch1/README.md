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
sample output:
```
python driver_batch1.py --model-path ~/dbrx-base/distributable/batch1 --prompt "To be, or not to be, that is the question: Whether 'tis nobler in" --max-tokens 64
==========
Prompt: To be, or not to be, that is the question: Whether 'tis nobler in
 the mind to suffer The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles, And by opposing end them? --Hamlet, Act 3, Scene 1

A question that was asked long ago by a troubled Danish prince is relevant to many of the problems that are encountered in
==========
Prompt: 20 tokens in 14.20202116703149 seconds = 1.408 t/s
Generation: 63 tokens in 108.62533266702667 seconds = 0.580 t/s
```
