export ARCHIVE_PATH="ckpts/$1/model_epoch_${2}.tar.gz"
if [ ! -f $ARCHIVE_PATH ];
then
  python3 utils/make_archive.py -s ckpts/$1 -n $2 || exit 1;
fi
NUM_EPOCHS=$4 FT_LANG=$3 RUN_NAME=$1_$2_$3_$5 UD_ROOT="/home/nlpmaster/ssd-1t/corpus/ud/ud-v2.2-preped/" python -W ignore run.py train training_config/fine-tune.jsonnet --include-package src -s ckpts/$1_$2_$3_$5
