RUN_NAME=$1 UD_ROOT="/home/nlpmaster/ssd-1t/corpus/ud/ud-v2.2-preped/" python -W ignore run.py train training_config/multilang_dependency_parser.jsonnet -s ckpts/$1 --include-package src
