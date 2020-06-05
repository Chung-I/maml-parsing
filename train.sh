RUN_NAME=$1 python -W ignore run.py train $2 -s ckpts/$1 --include-package src
