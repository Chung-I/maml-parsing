#export ARCHIVE_PATH="ckpts/$1"
if [[ $ENSEMBLE -eq "1" ]] ;
then
  script="ensemble.py"
else
  script="predict.py"
fi

test_file=${UD_ROOT}$2**-test.conllu
FT_LANG=$2 python -W ignore $script ckpts/$1 $test_file ckpts/$1/result.conllu --include-package src --cuda-device 0 --batch-size $3 || exit 1;
python utils/conll18_ud_eval.py $(ls ${UD_GT}"$(echo $2 | sed 's/[0-9]//g')"**-test.conllu) ckpts/$1/result.conllu -v > ckpts/$1/result.txt
  
test_file=${UD_GT}$2**-test.conllu
FT_LANG=$2 python -W ignore $script ckpts/$1 $test_file ckpts/$1/result-gt.conllu --include-package src --cuda-device 0 --batch-size $3 || exit 1;
#python utils/conll18_ud_eval.py $(ls ${UD_GT}"$(echo $2 | sed 's/[0-9]//g')"**-test.conllu) ckpts/$1/result-gt.conllu -v > ckpts/$1/result-gt.txt
python utils/error_analysis.py $(ls ${UD_GT}"$(echo $2 | sed 's/[0-9]//g')"**-test.conllu) ckpts/$1/result-gt.conllu -v > ckpts/$1/result-gt.txt
