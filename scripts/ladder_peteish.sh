./scripts/beaker/ladder_peteish-launch.sh 4 --model 190M --data olmoe-mix-0924 --length 1xC --name peteish --save_overwrite --device_batch_size 4 --batch_size_divisor 128
./scripts/beaker/ladder_peteish-launch.sh 8 --model 370M --data olmoe-mix-0924 --length 1xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./scripts/beaker/ladder_peteish-launch.sh 8 --model 600M --data olmoe-mix-0924 --length 1xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./scripts/beaker/ladder_peteish-launch.sh 8 --model 760M --data olmoe-mix-0924 --length 1xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./scripts/beaker/ladder_peteish-launch.sh 16 --model 1B --data olmoe-mix-0924 --length 1xC --name peteish --save_overwrite --device_batch_size 1 --batch_size_divisor 128

./scripts/beaker/ladder_peteish-launch.sh 4 --model 190M --data olmoe-mix-0924 --length 2xC --name peteish --save_overwrite --device_batch_size 4 --batch_size_divisor 128
./scripts/beaker/ladder_peteish-launch.sh 8 --model 370M --data olmoe-mix-0924 --length 2xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./scripts/beaker/ladder_peteish-launch.sh 8 --model 600M --data olmoe-mix-0924 --length 2xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./scripts/beaker/ladder_peteish-launch.sh 8 --model 760M --data olmoe-mix-0924 --length 2xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./scripts/beaker/ladder_peteish-launch.sh 16 --model 1B --data olmoe-mix-0924 --length 2xC --name peteish --save_overwrite --device_batch_size 1 --batch_size_divisor 128

./scripts/beaker/ladder_peteish-launch.sh 4 --model 190M --data olmoe-mix-0924 --length 5xC --name peteish --save_overwrite --device_batch_size 4 --batch_size_divisor 128
./scripts/beaker/ladder_peteish-launch.sh 8 --model 370M --data olmoe-mix-0924 --length 5xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./scripts/beaker/ladder_peteish-launch.sh 8 --model 600M --data olmoe-mix-0924 --length 5xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./scripts/beaker/ladder_peteish-launch.sh 8 --model 760M --data olmoe-mix-0924 --length 5xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./scripts/beaker/ladder_peteish-launch.sh 16 --model 1B --data olmoe-mix-0924 --length 5xC --name peteish --save_overwrite --device_batch_size 1 --batch_size_divisor 128

./scripts/beaker/ladder_peteish-launch.sh 4 --model 190M --data olmoe-mix-0924 --length 10xC --name peteish --save_overwrite --device_batch_size 4 --batch_size_divisor 128
./scripts/beaker/ladder_peteish-launch.sh 8 --model 370M --data olmoe-mix-0924 --length 10xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./scripts/beaker/ladder_peteish-launch.sh 8 --model 600M --data olmoe-mix-0924 --length 10xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./scripts/beaker/ladder_peteish-launch.sh 8 --model 760M --data olmoe-mix-0924 --length 10xC --name peteish --save_overwrite --device_batch_size 2 --batch_size_divisor 128
./scripts/beaker/ladder_peteish-launch.sh 16 --model 1B --data olmoe-mix-0924 --length 10xC --name peteish --save_overwrite --device_batch_size 1 --batch_size_divisor 128


./scripts/beaker/ladder_peteish-launch.sh 4 --model 190M --data olmoe-mix-0924 --length 10xC --name peteish-const --save_overwrite --device_batch_size 4 --batch_size_divisor 128 --alpha_f 1.0
./scripts/beaker/ladder_peteish-launch.sh 8 --model 370M --data olmoe-mix-0924 --length 10xC --name peteish-const --save_overwrite --device_batch_size 2 --batch_size_divisor 128 --alpha_f 1.0
./scripts/beaker/ladder_peteish-launch.sh 8 --model 600M --data olmoe-mix-0924 --length 10xC --name peteish-const --save_overwrite --device_batch_size 2 --batch_size_divisor 128 --alpha_f 1.0
./scripts/beaker/ladder_peteish-launch.sh 8 --model 760M --data olmoe-mix-0924 --length 10xC --name peteish-const --save_overwrite --device_batch_size 2 --batch_size_divisor 128 --alpha_f 1.0
./scripts/beaker/ladder_peteish-launch.sh 16 --model 1B --data olmoe-mix-0924 --length 10xC --name peteish-const --save_overwrite --device_batch_size 1 --batch_size_divisor 128 --alpha_f 1.0


./scripts/beaker/ladder_peteish-launch.sh 2 --model 190M --data olmoe-mix-0924 --length 1xC --name peteish-moreeval --save_overwrite --device_batch_size 4 --batch_size_divisor 64 --device_eval_batch_size 16
./scripts/beaker/ladder_peteish-launch.sh 2 --model 370M --data olmoe-mix-0924 --length 1xC --name peteish-moreeval --save_overwrite --device_batch_size 4 --batch_size_divisor 64 --device_eval_batch_size 16
./scripts/beaker/ladder_peteish-launch.sh 4 --model 760M --data olmoe-mix-0924 --length 1xC --name peteish-moreeval --save_overwrite --device_batch_size 2 --batch_size_divisor 64 --device_eval_batch_size 8
./scripts/beaker/ladder_peteish-launch.sh 8 --model 1B --data olmoe-mix-0924 --length 1xC --name peteish-moreeval --save_overwrite --device_batch_size 1 --batch_size_divisor 64 --device_eval_batch_size 4

./scripts/beaker/ladder_peteish-launch.sh 2 --model 190M --data olmoe-mix-0924 --length 2xC --name peteish-moreeval --save_overwrite --device_batch_size 4 --batch_size_divisor 64 --device_eval_batch_size 16
./scripts/beaker/ladder_peteish-launch.sh 2 --model 370M --data olmoe-mix-0924 --length 2xC --name peteish-moreeval --save_overwrite --device_batch_size 4 --batch_size_divisor 64 --device_eval_batch_size 16
./scripts/beaker/ladder_peteish-launch.sh 4 --model 760M --data olmoe-mix-0924 --length 2xC --name peteish-moreeval --save_overwrite --device_batch_size 2 --batch_size_divisor 64 --device_eval_batch_size 8
./scripts/beaker/ladder_peteish-launch.sh 8 --model 1B --data olmoe-mix-0924 --length 2xC --name peteish-moreeval --save_overwrite --device_batch_size 1 --batch_size_divisor 64 --device_eval_batch_size 4

./scripts/beaker/ladder_peteish-launch.sh 2 --model 190M --data olmoe-mix-0924 --length 5xC --name peteish-moreeval --save_overwrite --device_batch_size 4 --batch_size_divisor 64 --device_eval_batch_size 16 --save_interval 500 --eval_interval 500
./scripts/beaker/ladder_peteish-launch.sh 4 --model 370M --data olmoe-mix-0924 --length 5xC --name peteish-moreeval --save_overwrite --device_batch_size 2 --batch_size_divisor 64 --device_eval_batch_size 16 --save_interval 500 --eval_interval 500
./scripts/beaker/ladder_peteish-launch.sh 8 --model 760M --data olmoe-mix-0924 --length 5xC --name peteish-moreeval --save_overwrite --device_batch_size 1 --batch_size_divisor 64 --device_eval_batch_size 8 --save_interval 500 --eval_interval 500
./scripts/beaker/ladder_peteish-launch.sh 16 --model 1B --data olmoe-mix-0924 --length 5xC --name peteish-moreeval --save_overwrite --device_batch_size 1 --batch_size_divisor 128 --device_eval_batch_size 4 --save_interval 500 --eval_interval 500

./scripts/beaker/ladder_peteish-launch.sh 2 --model 190M --data olmoe-mix-0924 --length 10xC --name peteish-moreeval --save_overwrite --device_batch_size 4 --batch_size_divisor 64 --device_eval_batch_size 16 --save_interval 1000 --eval_interval 1000
./scripts/beaker/ladder_peteish-launch.sh 4 --model 370M --data olmoe-mix-0924 --length 10xC --name peteish-moreeval --save_overwrite --device_batch_size 2 --batch_size_divisor 64 --device_eval_batch_size 16 --save_interval 1000 --eval_interval 1000
./scripts/beaker/ladder_peteish-launch.sh 8 --model 760M --data olmoe-mix-0924 --length 10xC --name peteish-moreeval --save_overwrite --device_batch_size 1 --batch_size_divisor 64 --device_eval_batch_size 8 --save_interval 1000 --eval_interval 1000
./scripts/beaker/ladder_peteish-launch.sh 16 --model 1B --data olmoe-mix-0924 --length 10xC --name peteish-moreeval --save_overwrite --device_batch_size 1 --batch_size_divisor 64 --device_eval_batch_size 4 --save_interval 1000 --eval_interval 1000
