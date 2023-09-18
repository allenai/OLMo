# LLM Inference

## Compress

Run the following:

```
bash compression/run_olmo_quantization.sh /net/nfs.cirrascale/allennlp/akshitab/olmo-models/olmo-1b quantized_olmo-1b
```

## Run accuracy benchmark

Run the following:

```
bash eval/mmlu/eval_on_mmlu.sh quantized_olmo-1b /net/nfs.cirrascale/allennlp/akshitab/olmo-models/olmo-1b /net/nfs.cirrascale/allennlp/akshitab/data/mmlu eval_results
```

Output format:

```
Average accuracy 0.202 - math
Average accuracy 0.232 - health
Average accuracy 0.219 - physics
Average accuracy 0.270 - business
Average accuracy 0.198 - biology
Average accuracy 0.172 - chemistry
Average accuracy 0.267 - computer science
Average accuracy 0.204 - economics
Average accuracy 0.234 - engineering
Average accuracy 0.238 - philosophy
Average accuracy 0.236 - other
Average accuracy 0.233 - history
Average accuracy 0.177 - geography
Average accuracy 0.204 - politics
Average accuracy 0.225 - psychology
Average accuracy 0.250 - culture
Average accuracy 0.250 - law
Average accuracy 0.212 - STEM
Average accuracy 0.241 - humanities
Average accuracy 0.215 - social sciences
Average accuracy 0.238 - other (business, health, misc.)
Average accuracy: 0.229
```


## Run efficiency benchmark

Run the following:

```
cd efficiency
bash run_olmo_efficiency_benchmark.sh /net/nfs.cirrascale/allennlp/akshitab/olmo-models/olmo-1b quantized_olmo-1b
```

Output format:

```
Time Elapsed: 500.91 s
Max GPU memory usage:  2.09 GiB.
Average GPU power:  9.00e+01 W.
Average power:  2.04e+02 W.
Total energy:  7.49e-02 kWh.
CO2 emission:  6.35e-03 kg.
Throughput:  0.20 instances / s.
Throughput:  47.30 words / s.
Latency:  5009.10 ms / batch.
```
