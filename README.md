# M&M: Meta and Multitask are Stronger Together in Multilingual Dependency Parsing

This is the implementation of the anonymous EMNLP 2020 submission *M&M: Meta and Multitask are Stronger Together in Multilingual Dependency Parsing*.

## Requirements:
- conda
- \>= Python 3.6
- \>= PyTorch 1.3.0
- \>= allennlp
## Setting up the environment
1. Set up conda environment using the following command:
```=bash
conda env create -f environment.yml
```
2. Our codebase requires a specific commit of allennlp:
```=bash
conda activate maml-grammar
pip install git+git://github.com/allenai/allennlp.git@4b08f3ecb8e35c05f17c82145ad05e2bd36844bd
```

## Pre-training:
### Multi-task
```=bash
UD_GT="path/to/your/ud-treebanks-v2.2/**/" UD_ROOT="path/to/your/preprocessed/testset/" python -W ignore run.py train training_config/multi-adapter.jsonnet -s ckpts/<serialization_dir> --include-package src
```
### Reptile
```=bash
UD_GT="path/to/your/ud-treebanks-v2.x/**/" UD_ROOT="path/to/your/preprocessed/testset/" python -W ignore run.py train training_config/meta-adapter.jsonnet -s ckpts/<serialization_dir> --include-package src
```
## Zero-shot Transfer
`EPOCH_NUM`:  which pre-training epoch checkpoint to perform zero-shot transfer from.
`ZS_LANG`:  code of target transfer language (e.g. bxr, kmr, ..., etc.).
`SUFFIX`:  suffix of folder names storing results.
```=bash
UD_GT="path/to/your/ud-treebanks-v2.x/**/" UD_ROOT="path/to/your/preprocessed/testset/" bash zs-eval.sh <serialization_dir> $EPOCH_NUM $ZS_LANG 0 $SUFFIX
```

Results will be stored in `ckpts/<serialization_dir>_${EPOCH_NUM}_${ZS_LANG}_${SUFFIX}`.
## Fine-tuning
`EPOCH_NUM`:  which pre-training epoch checkpoint to perform fine-tuning from.
`ZS_LANG`:  code of target transfer language (e.g. bxr, kmr, ..., etc.).
`NUM_EPOCHS`: perform fine-tuning for how many epochs.
`SUFFIX`:  suffix of folder names storing results.
```=bash
UD_GT="path/to/your/ud-treebanks-v2.x/**/" UD_ROOT="path/to/your/preprocessed/testset/" BASE_MODEL=<serialization_dir> bash fine-tune.sh <serialization_dir> $EPOCH_NUM $FT_LANG $NUM_EPOCHS $SUFFIX
``` 

Results will be stored in `ckpts/<serialization_dir>_${EPOCH_NUM}_${FT_LANG}_${SUFFIX}`.
(`ckpts/<serialization_dir>_${EPOCH_NUM}_${FT_LANG}_ens_${SUFFIX}` for languages that require model ensemble; see paper for details.)
