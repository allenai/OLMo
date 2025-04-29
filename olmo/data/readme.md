

```
parallel --joblog progress_data.log --resume --progress -j1 python upload_to_hf.py --named-data-mix {} --local-dir /tmp/data/train --hf-repo-id allenai/DataDecide-data-recipes --num-download-workers 190 < /home/ubuntu/OLMo/olmo/data/ingredients.txt
```

```
python upload_to_hf.py --named-data-mix c4 --local-dir /tmp/data/train --hf-repo-id allenai/DataDecide-data-recipes --num-download-workers 190 --dolma-1-6-bypass
```