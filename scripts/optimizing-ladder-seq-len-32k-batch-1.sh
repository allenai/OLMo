scripts/beaker/ladder-launch.sh 1 normal --model 300M --data no_code --length 5xC --name no_code-seqlen-32k-batch-1 --save_overwrite
scripts/beaker/ladder-launch.sh 1 normal --model 300M --data no_math_no_code --length 5xC --name no_math_no_code-seqlen-32k-batch-1 --save_overwrite
scripts/beaker/ladder-launch.sh 1 normal --model 300M --data no_reddit --length 5xC --name no_reddit-seqlen-32k-batch-1 --save_overwrite
scripts/beaker/ladder-launch.sh 1 normal --model 300M --data no_flan --length 5xC --name no_flan-seqlen-32k-batch-1 --save_overwrite
scripts/beaker/ladder-launch.sh 1 normal --model 300M --data dolma17 --length 5xC --name dolma17-seqlen-32k-batch-1 --save_overwrite
scripts/beaker/ladder-launch.sh 1 normal --model 300M --data falcon --length 5xC --name falcon-seqlen-32k-batch-1 --save_overwrite
scripts/beaker/ladder-launch.sh 1 normal --model 300M --data falcon_and_cc --length 5xC --name falcon_and_cc-seqlen-32k-batch-1 --save_overwrite
scripts/beaker/ladder-launch.sh 1 normal --model 300M --data c4 --length 5xC --name c4-seqlen-32k-batch-1 --save_overwrite
scripts/beaker/ladder-launch.sh 1 normal --model 300M --data prox_fineweb_pro --length 5xC --name prox_fineweb_pro-seqlen-32k-batch-1 --save_overwrite --s3
scripts/beaker/ladder-launch.sh 1 normal --model 300M --data fineweb_edu_dedup --length 5xC --name fineweb_edu_dedup-seqlen-32k-batch-1 --save_overwrite --s3
scripts/beaker/ladder-launch.sh 1 normal --model 300M --data falcon_and_cc_eli5_oh_top10p --length 5xC --name falcon_and_cc_eli5_oh_top10p-seqlen-32k-batch-1 --save_overwrite --s3
scripts/beaker/ladder-launch.sh 1 normal --model 300M --data falcon_and_cc_eli5_oh_top20p --length 5xC --name falcon_and_cc_eli5_oh_top20p-seqlen-32k-batch-1 --save_overwrite --s3
scripts/beaker/ladder-launch.sh 1 normal --model 300M --data falcon_and_cc_og_eli5_oh_top10p --length 5xC --name falcon_and_cc_og_eli5_oh_top10p-seqlen-32k-batch-1 --save_overwrite --s3
scripts/beaker/ladder-launch.sh 1 normal --model 300M --data falcon_and_cc_tulu_qc_top10 --length 5xC --name falcon_and_cc_tulu_qc_top10-seqlen-32k-batch-1 --save_overwrite --s3
scripts/beaker/ladder-launch.sh 1 normal --model 300M --data dolma-v1-6-and-sources-baseline --length 5xC --name dolma-v1-6-and-sources-baseline-seqlen-32k-batch-1 --save_overwrite --s3
scripts/beaker/ladder-launch.sh 1 normal --model 300M --data DCLM-baseline --length 5xC --name DCLM-baseline-seqlen-32k-batch-1 --save_overwrite
scripts/beaker/ladder-launch.sh 1 normal --model 300M --data dolma17-75p-DCLM-baseline-25p --length 5xC --name dolma17-75p-DCLM-baseline-25p-seqlen-32k-batch-1 --save_overwrite
scripts/beaker/ladder-launch.sh 1 normal --model 300M --data dolma17-50p-DCLM-baseline-50p --length 5xC --name dolma17-50p-DCLM-baseline-50p-seqlen-32k-batch-1 --save_overwrite
scripts/beaker/ladder-launch.sh 1 normal --model 300M --data dolma17-25p-DCLM-baseline-75p --length 5xC --name dolma17-25p-DCLM-baseline-75p-seqlen-32k-batch-1 --save_overwrite
