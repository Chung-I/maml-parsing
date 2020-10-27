if [[ $ENSEMBLE -eq "1" ]] ;
then
  script="ensemble.py"
else
  script="predict.py"
fi

test_file=${UD_ROOT}$2**-test.conllu
FT_LANG=$2 python -W ignore $script $1 $test_file $1/result.conllu --include-package src --cuda-device 0 --batch-size $3 || exit 1;
python utils/conll18_ud_eval.py $(ls ${UD_GT}"$(echo $2 | sed 's/[0-9]//g')"**-test.conllu) $1/result.conllu -v > $1/result.txt
  
plain_lang="$(echo $2 | sed 's/[0-9]//g')"
test_file=$(ls ${UD_GT}${plain_lang}**-test.conllu)
echo $test_file
FT_LANG=${plain_lang} python -W ignore $script $1 $test_file $1/result-gt.conllu --include-package src --cuda-device 0 --batch-size $3 || exit 1;
python utils/error_analysis.py $test_file $1/result-gt.conllu -v > $1/result-gt.txt
