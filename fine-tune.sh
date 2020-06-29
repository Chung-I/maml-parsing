CV=""
export ARCHIVE_PATH="ckpts/$1/model_epoch_${2}.tar.gz"
if [ ! -f $ARCHIVE_PATH ];
then
  python3 utils/make_archive.py -s ckpts/$1 -n $2 -m training_config/model.json || exit 1;
fi
if grep -w $3 data/ensemble_langs.txt;
then
  for file in $UD_ROOT${3}_*-*-*-train.conllu;
  do
    echo $file
    CV_NUM=$(echo $(basename $file) | awk -F'-' '{print $2}')
    echo $CV_NUM
    CV="_*-$CV_NUM-" NUM_EPOCHS=$4 FT_LANG=$3 RUN_NAME=$1_$2_$3_$5 python -W ignore run.py train training_config/fine-tune.jsonnet --include-package src -s ckpts/$1_$2_$3_cv${CV_NUM}_$5
  done
else
  CV="" NUM_EPOCHS=$4 FT_LANG=$3 RUN_NAME=$1_$2_$3_$5 python -W ignore run.py train training_config/fine-tune.jsonnet --include-package src -s ckpts/$1_$2_$3_$5
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
  python3 tools/make_ens_config.py "ckpts/$1_$2_$3_cv*_$5/" ckpts/$1_$2_$3_ens_$5
  ENSEMBLE="1" bash test.sh $1_$2_$3_ens_$5 $3 8
  for file in $UD_ROOT${3}_*-*-*-train.conllu;
  do
    echo $file
    CV_NUM=$(echo $(basename $file) | awk -F'-' '{print $2}')
    echo $CV_NUM
    rm ckpts/$1_$2_$3_cv${CV_NUM}_$5/*.th
  done
else
  bash test.sh $1_$2_$3_$5 $3 8
  rm ckpts/$1_$2_$3_$5/*.th
fi
