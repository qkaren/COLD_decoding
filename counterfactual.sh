#!/bin/bash

## Counterfactual

python3 cold_decoding.py \
	--seed 12 \
	--mode counterfactual_langevin \
	--pretrained_model gpt2-xl \
	--init-temp 1 \
    --length 20 \
	--max-length 50 \
	--num-iters 2000 \
	--min-iters 10 \
	--constraint-weight 0.2 \
    --counterfactual-max-ngram 3 \
	--stepsize 0.1 \
	--noise-iters 1 \
	--win-anneal-iters 1000 \
	--start 0 \
	--end 5 \
	--lr-nll-portion 0.9 \
    --topk 5 \
    --output-lgt-temp 1 \
	--verbose \
    --straight-through  \
	--large-noise-iters 50,200,500 \
	--large_gs_std 0.5,0.1,0.05  \
	--input-file "./data/counterfactual/dev_data.json" \
	--output-dir "./data/counterfactual/" \
	--stepsize-ratio 1  \
    --batch-size 32 \
    --print-every 200

