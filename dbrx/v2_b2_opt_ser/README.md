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
sample output:
```
python driver_ser.py --model-path ~/dbrx-base/distributable/batch2 --prompt "To be, or not to be, that is the question: Whether 'tis nobler in" --max-tokens 100
==========
Prompt: To be, or not to be, that is the question: Whether 'tis nobler in
 the mind to suffer The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles, And by opposing end them? --Hamlet, Act 3, Scene 1

A question that was asked long ago by a troubled Danish prince is relevant to many of the problems that are encountered in the development of advanced software systems. Whether it is a better strategy to suffer the known problems of our current methods or to take the risk of adopting an untried alternative is a
==========
Prompt: 20 tokens in 6.0102464579977095 seconds = 3.328 t/s
Generation: 99 tokens in 30.94716600002721 seconds = 3.199 t/s
```
