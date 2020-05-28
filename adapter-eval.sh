export ARCHIVE_PATH="ckpts/$1/model_epoch_${2}.tar.gz"
#CV="" NUM_EPOCHS=$4 FT_LANG=$3 RUN_NAME=$1_$2_$3_$5 python -W ignore run.py train training_config/adapter-ft.jsonnet --include-package src -s ckpts/$1_$2_$3_$5
mkdir -p ckpts/$1_$2_$3_$5
CV="" NUM_EPOCHS=$4 FT_LANG=$3 RUN_NAME=$1_$2_$3_$5 python3 tools/manifest.py training_config/adapter-ft.jsonnet ckpts/$1_$2_$3_$5/config.json
cp ckpts/$1/model_state_epoch_$2.th ckpts/$1_$2_$3_$5/best.th
cp -r ckpts/$1/vocabulary ckpts/$1_$2_$3_$5
bash dev.sh $1_$2_$3_$5 $3 8 train
bash test.sh $1_$2_$3_$5 $3 8
rm ckpts/$1_$2_$3_$5/*.th
