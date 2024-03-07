#!/bin/bash
for i in 1024 2048 4096 8192 16384 32768 65536
do
    for j in {1..20}
    do
        # echo "iter: $i" >> output.txt
        # python mlx_test.py -t 100 -n 20 -l >> output.txt
        python mlx_test.py --data-size 100 --mat-dim $i --n-samples 20 --log-path logs_${i}_batch.csv
        python mlx_test.py --data-size 100 --mat-dim $i --load-sep --n-samples 20 --log-path logs_${i}_sep.csv
    done
done