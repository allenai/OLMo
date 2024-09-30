./scripts/beaker/ladder-launch.sh 4 --model 150M --data dolma17 --length 1xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 4 --model 300M --data dolma17 --length 1xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 530M --data dolma17 --length 1xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 750M --data dolma17 --length 1xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 1B --data dolma17 --length 1xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 3B --data dolma17 --length 1xC --name amberish-5shot --s3 --save_overwrite

./scripts/beaker/ladder-launch.sh 4 --model 150M --data dolma17 --length 2xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 4 --model 300M --data dolma17 --length 2xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 530M --data dolma17 --length 2xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 750M --data dolma17 --length 2xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 1B --data dolma17 --length 2xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 3B --data dolma17 --length 2xC --name amberish-5shot --s3 --save_overwrite

./scripts/beaker/ladder-launch.sh 4 --model 150M --data dolma17 --length 5xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 4 --model 300M --data dolma17 --length 5xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 530M --data dolma17 --length 5xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 750M --data dolma17 --length 5xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 1B --data dolma17 --length 5xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 3B --data dolma17 --length 5xC --name amberish-5shot --s3 --save_overwrite

./scripts/beaker/ladder-launch.sh 4 --model 150M --data dolma17 --length 10xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 4 --model 300M --data dolma17 --length 10xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 530M --data dolma17 --length 10xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 750M --data dolma17 --length 10xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 1B --data dolma17 --length 10xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 3B --data dolma17 --length 10xC --name amberish-5shot --s3 --save_overwrite

./scripts/beaker/ladder-launch.sh 4 --model 150M --data dolma17 --length 10xC --name amberish-const --s3 --save_overwrite --alpha_f 1.0
./scripts/beaker/ladder-launch.sh 4 --model 300M --data dolma17 --length 10xC --name amberish-const --s3 --save_overwrite --alpha_f 1.0
./scripts/beaker/ladder-launch.sh 8 --model 530M --data dolma17 --length 10xC --name amberish-const --s3 --save_overwrite --alpha_f 1.0
./scripts/beaker/ladder-launch.sh 8 --model 750M --data dolma17 --length 10xC --name amberish-const --s3 --save_overwrite --alpha_f 1.0
./scripts/beaker/ladder-launch.sh 8 --model 1B --data dolma17 --length 10xC --name amberish-const --s3 --save_overwrite --alpha_f 1.0

./scripts/beaker/ladder-launch.sh 4 --model 150M --data dolma17 --length 1xC --name amberish-const-decay --s3 --save_overwrite --scheduler_type wsd --load_path s3://ai2-llm/checkpoints/OLMo-ladder/amberish-const-150M-10xC/step2800-unsharded
./scripts/beaker/ladder-launch.sh 4 --model 300M --data dolma17 --length 1xC --name amberish-const-decay --s3 --save_overwrite --scheduler_type wsd --load_path s3://ai2-llm/checkpoints/OLMo-ladder/amberish-const-300M-10xC/step3400-unsharded
./scripts/beaker/ladder-launch.sh 8 --model 530M --data dolma17 --length 1xC --name amberish-const-decay --s3 --save_overwrite --scheduler_type wsd --load_path s3://ai2-llm/checkpoints/OLMo-ladder/amberish-const-530M-10xC/step4200-unsharded
./scripts/beaker/ladder-launch.sh 8 --model 750M --data dolma17 --length 1xC --name amberish-const-decay --s3 --save_overwrite --scheduler_type wsd --load_path s3://ai2-llm/checkpoints/OLMo-ladder/amberish-const-750M-10xC/step4600-unsharded
./scripts/beaker/ladder-launch.sh 8 --model 1B --data dolma17 --length 1xC --name amberish-const-decay --s3 --save_overwrite --scheduler_type wsd --load_path s3://ai2-llm/checkpoints/OLMo-ladder/amberish-const-1B-10xC/step5200-unsharded

./scripts/beaker/ladder-launch.sh 4 --model 150M --data dolma17 --length 2xC --name amberish-const-decay --s3 --save_overwrite --scheduler_type wsd --load_path s3://ai2-llm/checkpoints/OLMo-ladder/amberish-const-150M-10xC/step6600-unsharded
./scripts/beaker/ladder-launch.sh 4 --model 300M --data dolma17 --length 2xC --name amberish-const-decay --s3 --save_overwrite --scheduler_type wsd --load_path s3://ai2-llm/checkpoints/OLMo-ladder/amberish-const-300M-10xC/step8000-unsharded
./scripts/beaker/ladder-launch.sh 8 --model 530M --data dolma17 --length 2xC --name amberish-const-decay --s3 --save_overwrite --scheduler_type wsd --load_path s3://ai2-llm/checkpoints/OLMo-ladder/amberish-const-530M-10xC/step10000-unsharded
./scripts/beaker/ladder-launch.sh 8 --model 750M --data dolma17 --length 2xC --name amberish-const-decay --s3 --save_overwrite --scheduler_type wsd --load_path s3://ai2-llm/checkpoints/OLMo-ladder/amberish-const-750M-10xC/step11000-unsharded
./scripts/beaker/ladder-launch.sh 8 --model 1B --data dolma17 --length 2xC --name amberish-const-decay --s3 --save_overwrite --scheduler_type wsd --load_path s3://ai2-llm/checkpoints/OLMo-ladder/amberish-const-1B-10xC/step12000-unsharded

./scripts/beaker/ladder-launch.sh 4 --model 150M --data dolma17 --length 5xC --name amberish-const-decay --s3 --save_overwrite --scheduler_type wsd --load_path s3://ai2-llm/checkpoints/OLMo-ladder/amberish-const-150M-10xC/step18000-unsharded
./scripts/beaker/ladder-launch.sh 4 --model 300M --data dolma17 --length 5xC --name amberish-const-decay --s3 --save_overwrite --scheduler_type wsd --load_path s3://ai2-llm/checkpoints/OLMo-ladder/amberish-const-300M-10xC/step21600-unsharded
./scripts/beaker/ladder-launch.sh 8 --model 530M --data dolma17 --length 5xC --name amberish-const-decay --s3 --save_overwrite --scheduler_type wsd --load_path s3://ai2-llm/checkpoints/OLMo-ladder/amberish-const-530M-10xC/step27400-unsharded
./scripts/beaker/ladder-launch.sh 8 --model 750M --data dolma17 --length 5xC --name amberish-const-decay --s3 --save_overwrite --scheduler_type wsd --load_path s3://ai2-llm/checkpoints/OLMo-ladder/amberish-const-750M-10xC/step30200-unsharded
./scripts/beaker/ladder-launch.sh 8 --model 1B --data dolma17 --length 5xC --name amberish-const-decay --s3 --save_overwrite --scheduler_type wsd --load_path s3://ai2-llm/checkpoints/OLMo-ladder/amberish-const-1B-10xC/step32800-unsharded

./scripts/beaker/ladder-launch.sh 4 --model 150M --data dolma17 --length 10xC --name amberish-const-decay --s3 --save_overwrite --scheduler_type wsd --load_path s3://ai2-llm/checkpoints/OLMo-ladder/amberish-const-150M-10xC/step37200-unsharded
./scripts/beaker/ladder-launch.sh 4 --model 300M --data dolma17 --length 10xC --name amberish-const-decay --s3 --save_overwrite --scheduler_type wsd --load_path s3://ai2-llm/checkpoints/OLMo-ladder/amberish-const-300M-10xC/step44600-unsharded
./scripts/beaker/ladder-launch.sh 8 --model 530M --data dolma17 --length 10xC --name amberish-const-decay --s3 --save_overwrite --scheduler_type wsd --load_path s3://ai2-llm/checkpoints/OLMo-ladder/amberish-const-530M-10xC/step56200-unsharded
./scripts/beaker/ladder-launch.sh 8 --model 750M --data dolma17 --length 10xC --name amberish-const-decay --s3 --save_overwrite --scheduler_type wsd --load_path s3://ai2-llm/checkpoints/OLMo-ladder/amberish-const-750M-10xC/step61800-unsharded
./scripts/beaker/ladder-launch.sh 8 --model 1B --data dolma17 --length 10xC --name amberish-const-decay --s3 --save_overwrite --scheduler_type wsd --load_path s3://ai2-llm/checkpoints/OLMo-ladder/amberish-const-1B-10xC/step67600-unsharded

./scripts/beaker/ladder-launch.sh 4 --model 150M --data dolma17_flan_sep_rulebased --length 2xC --name amberish-5shot-olddata --s3 --save_overwrite
