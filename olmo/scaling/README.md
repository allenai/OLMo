

## muP implementation for OLMo

### Changes made to the model

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

```commandline
python olmo/scaling/mup_coord_check.py test_fixtures/mup_train_tiny.yaml \
    --coord_check \
    --lr 0.01 \
    --load_base_shapes tiny-olmo-base-shapes.bsh \
    --coord_check_nsteps 1 \
    --coord_check_nseeds 1
```
