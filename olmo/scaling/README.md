

## muP implementation for OLMo

### Changes made to the model class

1. Replace the readout layer with MuReadout or MuSharedReadout
2. Replace torch.nn.init.normal_ with mup.init.normal_
3. Scale attention weights by 1/d instead of q/sqrt(d).

Other updates: Added input, output, attn multipliers.

#### Implementation references

1. muP [Transformer](https://github.com/microsoft/mup/blob/main/examples/Transformer/model.py) example.
2. mutransformers [gpt2](https://github.com/microsoft/mutransformers/blob/main/mutransformers/models/gpt2/modeling_gpt2.py) example.
3. LLM360 [CrystalCoder](https://huggingface.co/LLM360/CrystalCoder/blob/main/modeling_crystalcoder.py) example.

### Running coord check

1. Get sample data (from R2) for running coord check:

```commandline
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
aws s3 cp --endpoint-url=https://a198dc34621661a1a66a02d6eb7c4dc3.r2.cloudflarestorage.com  s3://olmo-data/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-010-00002.npy test_fixtures/mup-sample-data/
```

2. Save base shapes:

```commandline
 python olmo/scaling/mup_coord_check.py test_fixtures/mup_train_tiny.yaml --save_base_shapes tiny-olmo-base-shapes.bsh
```

3. Run coord check:

Temporary workaround for olmo's distributed code:
```commandline
export RANK=0
```

```commandline
python olmo/scaling/mup_coord_check.py test_fixtures/mup_train_tiny.yaml
    --coord_check \
    --lr 0.002  \
   --load_base_shapes tiny-olmo-base-shapes.bsh   \
   --coord_check_nsteps 5    \
   --coord_check_nseeds 3 \
   --cuda \
   --batch_size 2 \
   --optimizer muadamw
```
