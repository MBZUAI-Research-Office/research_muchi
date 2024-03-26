#!/bin/bash
for mat_dim in 4096 8192 16384
do
    for i in {1..20}
    do
        echo "iter $i"
        python mlx_test.py --data-size 100 --mat-dim ${mat_dim} --weights-path test_batch_100_${mat_dim}_weights.npy --n-samples 100 --output-path test_batch_${mat_dim}_${i}_output --log-path test_batch_${mat_dim}_reboot_logs.csv
        python mlx_test.py --data-size 100 --mat-dim ${mat_dim} --weights-path test_sep_100_${mat_dim}_weights.npz --load-sep --n-samples 100 --output-path test_sep_${mat_dim}_${i}_output --log-path test_sep_${mat_dim}_reboot_logs.csv
        python perf_test_validation.py --output-0 test_batch_${mat_dim}_${i}_output.npy --output-1 test_sep_${mat_dim}_${i}_output.npy
        printf "\n"
    done
done