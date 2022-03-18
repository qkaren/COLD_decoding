#!/bin/bash

## Abductive

python3 cold_decoding.py  \
	--seed 12 \
	--mode abductive_langevin \
	--pretrained_model gpt2-xl \
	--init-temp 1 \
    --length 10 \
	--max-length 40 \
	--num-iters 2000 \
	--min-iters 1000 \
	--constraint-weight 0.5 \
    --abductive-c2-weight 0.05 \
	--stepsize 0.1 \
	--noise-iters 1 \
	--win-anneal-iters 1000 \
	--start 0 \
	--end 5 \
	--lr-nll-portion 0.6 \
    --topk 2 \
    --output-lgt-temp 1 \
	--verbose \
    --straight-through  \
	--large-noise-iters 50,500,1000,1500 \
	--large_gs_std 1,0.5,0.1,0.05  \
	--input-file "./data/abductive/small_data.json" \
	--output-dir "./data/abductive/" \
	--stepsize-ratio 1  \
    --batch-size 16 \
    --print-every 200
