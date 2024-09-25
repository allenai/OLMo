./scripts/beaker/ladder-launch.sh 4 --model 150M --data dolma17 --length 1xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 4 --model 300M --data dolma17 --length 1xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 530M --data dolma17 --length 1xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 750M --data dolma17 --length 1xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 1B --data dolma17 --length 1xC --name amberish-5shot --s3 --save_overwrite

./scripts/beaker/ladder-launch.sh 4 --model 150M --data dolma17 --length 2xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 4 --model 300M --data dolma17 --length 2xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 530M --data dolma17 --length 2xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 750M --data dolma17 --length 2xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 1B --data dolma17 --length 2xC --name amberish-5shot --s3 --save_overwrite

./scripts/beaker/ladder-launch.sh 4 --model 150M --data dolma17 --length 5xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 4 --model 300M --data dolma17 --length 5xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 530M --data dolma17 --length 5xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 750M --data dolma17 --length 5xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 1B --data dolma17 --length 5xC --name amberish-5shot --s3 --save_overwrite

./scripts/beaker/ladder-launch.sh 4 --model 150M --data dolma17 --length 10xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 4 --model 300M --data dolma17 --length 10xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 530M --data dolma17 --length 10xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 750M --data dolma17 --length 10xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 1B --data dolma17 --length 10xC --name amberish-5shot --s3 --save_overwrite

./scripts/beaker/ladder-launch.sh 4 --model 150M --data dolma17 --length 10xC --name amberish-const --s3 --save_overwrite --alpha_f 1.0
./scripts/beaker/ladder-launch.sh 4 --model 300M --data dolma17 --length 10xC --name amberish-const --s3 --save_overwrite --alpha_f 1.0
./scripts/beaker/ladder-launch.sh 8 --model 530M --data dolma17 --length 10xC --name amberish-const --s3 --save_overwrite --alpha_f 1.0
./scripts/beaker/ladder-launch.sh 8 --model 750M --data dolma17 --length 10xC --name amberish-const --s3 --save_overwrite --alpha_f 1.0
./scripts/beaker/ladder-launch.sh 8 --model 1B --data dolma17 --length 10xC --name amberish-const --s3 --save_overwrite --alpha_f 1.0

./scripts/beaker/ladder-launch.sh 8 --model 3B --data dolma17 --length 1xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 3B --data dolma17 --length 2xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 3B --data dolma17 --length 5xC --name amberish-5shot --s3 --save_overwrite
./scripts/beaker/ladder-launch.sh 8 --model 3B --data dolma17 --length 10xC --name amberish-5shot --s3 --save_overwrite

./scripts/beaker/ladder-launch.sh 4 --model 150M --data dolma17_flan_sep_rulebased --length 2xC --name amberish-5shot-olddata --s3 --save_overwrite
