export ARCHIVE_PATH="ckpts/$1/model.tar.gz"
export UD_ROOT="/home/nlpmaster/ssd-1t/corpus/ud/ud-v2.2-preped/"
test_file=${UD_ROOT}$2-ud-test.conllu
FT_LANG=$2 python -W ignore predict.py $ARCHIVE_PATH $test_file ckpts/$1/result.conllu --include-package src --cuda-device 1 --batch-size $3 || exit 1;
python utils/conll18_ud_eval.py $(ls /home/nlpmaster/ssd-1t/corpus/ud/ud-treebanks-v2.2/*/$2-ud-test.conllu) ckpts/$1/result.conllu -v > ckpts/$1/result.txt
