#export ARCHIVE_PATH="ckpts/$1"
if [[ $ENSEMBLE -eq "1" ]] ;
then
  script="ensemble.py"
else
  script="predict.py"
fi

dev_file=${UD_GT}$2**-$4.conllu
out_file=ckpts/$1/$4-result.conllu
FT_LANG=$2 python -W ignore $script ckpts/$1 $dev_file $out_file --include-package src --cuda-device 0 --batch-size $3 || exit 1;
python utils/conll18_ud_eval.py $dev_file $out_file -v > ckpts/$1/$4-result.txt
