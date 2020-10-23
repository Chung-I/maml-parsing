CV=""
if [[ -z "${BS}" ]]; then
  MY_BS=8
else
  MY_BS=${BS}
fi

export ARCHIVE_PATH="$SCKPT/$1/model_epoch_${2}.tar.gz"

if [[ -z "${FT_SCRIPT}" ]]; then
  FT_SCRIPT="fine-tune.jsonnet"
else
  FT_SCRIPT=${FT_SCRIPT}
fi

if [ ! -f $ARCHIVE_PATH ];
then
  python3 utils/make_archive.py -s $SCKPT/$1 -n $2 -m training_config/model.json || exit 1;
fi

if grep -w $3 data/ensemble_langs.txt;
then
  for file in $UD_ROOT${3}_*-*-*-train.conllu;
  do
    echo $file
    CV_NUM=$(echo $(basename $file) | awk -F'-' '{print $2}')
    echo $CV_NUM
    MY_BS=$MY_BS CV="_*-$CV_NUM-" NUM_EPOCHS=$4 FT_LANG=$3 RUN_NAME=$1_$2_$3_$5 python -W ignore run.py train training_config/$FT_SCRIPT --include-package src -s $TCKPT/$1_$2_$3_cv${CV_NUM}_$5
    BASE_MODEL=$SCKPT/$1 python3 utils/make_full_model.py -s $TCKPT/$1_$2_$3_cv${CV_NUM}_$5 || exit 1;
  done
else
  MY_BS=$MY_BS CV="" NUM_EPOCHS=$4 FT_LANG=$3 RUN_NAME=$1_$2_$3_$5 python -W ignore run.py train training_config/$FT_SCRIPT --include-package src -s $TCKPT/$1_$2_$3_$5
  BASE_MODEL=$SCKPT/$1 python3 utils/make_full_model.py -s $TCKPT/$1_$2_$3_$5 || exit 1;
fi
if grep -w $3 data/ensemble_langs.txt;
then
  for file in $UD_ROOT${3}_*-*-*-train.conllu;
  do
    echo $file
    CV_NUM=$(echo $(basename $file) | awk -F'-' '{print $2}')
    echo $CV_NUM
    bash test.sh $1_$2_$3_cv${CV_NUM}_$5 $3 8
  done
  python3 tools/make_ens_config.py "$TCKPT/$1_$2_$3_cv*_$5/" $TCKPT/$1_$2_$3_ens_$5
  ENSEMBLE="1" bash test.sh $1_$2_$3_ens_$5 $3 8
  for file in $UD_ROOT${3}_*-*-*-train.conllu;
  do
    echo $file
    CV_NUM=$(echo $(basename $file) | awk -F'-' '{print $2}')
    echo $CV_NUM
    rm $TCKPT/$1_$2_$3_cv${CV_NUM}_$5/training_state_epoch_*.th
  done
else
  bash test.sh $1_$2_$3_$5 $3 8
  rm $TCKPT/$1_$2_$3_$5/*.th 
fi
