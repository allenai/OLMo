# Wikipedia Ablation Example

Run all following commands from root of this repository.

## Step 1: Run Taggers

Install filter code:

```shell
# make sure to install an conda if on a bare machine
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge

# if on linux, make sure gcc and protobuf are installed, e.g.
sudo apt install build-essential protobuf-compiler -y

# now install the filters
pip install pretrain_data/filters

# if on macOS, also run
python -m smashed.utils.install_blingfire_macos
```

Add tags:

```shell
ai2_llm_filters \
    -d wikipedia/v0 \
    -n abl0 \
    -t random_number_v1 \
        cld2_en_paragraph_with_doc_score_v2 \
        ft_lang_id_en_paragraph_with_doc_score_v2 \
        char_length_with_paragraphs_v1 \
        whitespace_tokenizer_with_paragraphs_v1 \
    -p 96   # run on 96 cores
```

## Step 2: Run Mixer

Compile and install mixer

```shell
cd pretrain_data/mixer
make build-tools    # will install rust and tools to build the mixer
make mixer          # will build the mixer; available at ./target/release/mixer
```

Now run mixer with `mixer_config.json`:

```shell
MIXER_BIN="pretrain_data/mixer/target/release/mixer"
$MIXER_BIN \
    examples/wikipedia_ablation/mixer_config.json
```

You can check out the mixer config to see how it works. In particular, it applies four operations:

- Include all documents with length less than 100,000 whitespace-separated words:
    ```json
    "include": [
        "$.attributes[?(@.abl0__whitespace_tokenizer_with_paragraphs_v1__document[0][2] < 100000)]"
    ]
    ```
- Remove any document that is shorter than 50 words:
    ```json
    "exclude": [
        "$.attributes[?(@.abl0__whitespace_tokenizer_with_paragraphs_v1__document[0][2] < 50)]",
        ...
    ]
- Remove any document whose total English cld2 score is below 0.5:
    ```json
    "exclude": [
        ...,
        "$.attributes[?(@.abl0__ft_lang_id_en_paragraph_with_doc_score_v2__doc_en[0][2] <= 0.5)]"
    ]
    ```
- Replace paragraphs whose not-English cld2 socre is below 0.9 in a document with an empty string
    ```json
    "span_replacement": [
        {
            "span": "$.attributes.abl0__cld2_en_paragraph_with_doc_score_v2__not_en",
            "min_score": 0.1,
            "replacement": ""
        }
    ]
    ```
