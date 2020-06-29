export ARCHIVE_PATH="ckpts/$1/model_epoch_${2}.tar.gz"
mkdir -p ckpts/$1_$2_$3_$5
CV="" NUM_EPOCHS=$4 FT_LANG=$3 RUN_NAME=$1_$2_$3_$5 python3 tools/manifest.py training_config/ft.jsonnet ckpts/$1_$2_$3_$5/config.json
cp ckpts/$1/model_state_epoch_$2.th ckpts/$1_$2_$3_$5/best.th
cp -r ckpts/$1/vocabulary ckpts/$1_$2_$3_$5

if [[ -z "${RUN_TRAIN}" ]]; then
  MY_RUN_TRAIN="true"
else
  MY_RUN_TRAIN=${RUN_TRAIN}
fi

if [[ $MY_RUN_TRAIN == "true" ]];
then
  bash dev.sh $1_$2_$3_$5 $3 8 train
fi
bash dev.sh $1_$2_$3_$5 $3 8 dev
bash test.sh $1_$2_$3_$5 $3 8
rm ckpts/$1_$2_$3_$5/*.th
