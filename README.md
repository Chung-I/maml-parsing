# Meta-learning for Low-Resource Dependency Parsing

## Data
All of the experiments were conducted on Universal Dependencies:
- [corpus main page](https://universaldependencies.org/)
- [data download page](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3226)
- [Universal Dependencies v2.2](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2837/ud-treebanks-v2.2.tgz?sequence=1&isAllowed=y) (we use 46 languages in v2.2 for training; see [data/train_langs.txt](data/train_langs.txt))
- [Universal Dependencies v2.5](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3105/ud-treebanks-v2.5.tgz?sequence=1&isAllowed=y) (we use 8 treebanks in v2.5 for testing; see [data/test_tbs-v2.5.txt](data/test_tbs-v2.5.txt))

## Setting up the environment
1. Set up conda environment:
```=bash
conda create -n maml-parsing python=3.6
conda activate maml-parsing
```
2. Install python package requirements:
```=bash
pip install -r requirements.txt
```

## Pre-training:

- `UD_GT`: Root path of ground truth universal dependencies treebank files used for evaluation.
- `UD_ROOT`: Root path of treebank files used for training. For scenarios that use ground truth universal dependencies treebank files for training, simply set it the same as `UD_GT`. For those who would like to use their own POS taggers(e.g. for comparison with [paper](https://www.aclweb.org/anthology/K18-2016.pdf)) as input features for training, put all pos-tagged conllu files in a singler folder and set `UD_ROOT` to it.
- `CONFIG_NAME`: json file storing training configuration such as dataset paths, model hyperparameter settings, training schedule, etc. See [delexicalized parsing models](#delexicalized-parsing-models) and [lexicalized parsing models](#lexicalized-parsing-models) for examples of configuration files to choose from.
- **Normal usage**: Simply extract [Universal Dependencies v2.2](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2837/ud-treebanks-v2.2.tgz?sequence=1&isAllowed=y) to some `folder`, then set `UD_GT="folder/**/"` and `UD_ROOT="folder/**/"`.
```=bash
UD_GT="path/to/your/ud-treebanks-v2.2/**/" UD_ROOT="path/to/your/pos-tagged/conllu-files/" python -W ignore run.py train $CONFIG_NAME -s <serialization_dir> --include-package src
```

### delexicalized parsing models:
- Multi-task baseline
    - `CONFIG_NAME=`[training_config/multi-pos.jsonnet](training_config/multi-pos.jsonnet)
    - pre-trained model: [multi-pos.tar.gz](https://drive.google.com/file/d/1FxDu5yrRHCw70sakJwXfOuZuU5fMfeDC/view?usp=sharing)
- MAML
    - `CONFIG_NAME=`[training_config/maml-pos.jsonnet](training_config/maml-pos.jsonnet)
    - pre-trained model: [maml-pos.tar.gz](https://drive.google.com/file/d/1_EVTZawK8ull66PjqSC-2t3-T026INaW/view?usp=sharing)
- FOMAML
    - `CONFIG_NAME=`[training_config/fomaml-pos.jsonnet](training_config/fomaml-pos.jsonnet)
    - pre-trained model: [fomaml-pos.tar.gz](https://drive.google.com/file/d/1TlOClQ5UAYtKwgKkw5d7OPpDdRZpojgT/view?usp=sharing)
- Reptile:
    - `CONFIG_NAME=`[training_config/reptile-pos.jsonnet](training_config/reptile-pos.jsonnet)
    - pre-trained model: [reptile-pos.tar.gz](https://drive.google.com/file/d/16X2ouqFfbd-FLDi7kLza2X7hUWcf_Lpd/view?usp=sharing)

### lexicalized parsing models:
- Multi-task baseline
    - `CONFIG_NAME=`[training_config/multi-lex.jsonnet](training_config/multi-lex.jsonnet)
    - pretrained model: [multi-lex.tar.gz](https://drive.google.com/file/d/1D6hHFuQey6m9gIIDeo8CkEglMi7P_HwX/view?usp=sharing)
- Reptile
    - `CONFIG_NAME=`[training_config/reptile-lex.jsonnet](training_config/reptile-lex.jsonnet)
    - pretrained model (inner step K=2): [reptile-lex-K2.tar.gz](https://drive.google.com/file/d/1pc0xTWtZhDAWFE5XhOaBNJxJymFuUL9m/view?usp=sharing)
    - pretrained model (inner step K=4): [reptile-lex-K4.tar.gz](https://drive.google.com/file/d/1-bmNc6JTCh5v0zfDpxuQEJ2e0XZPDrsD/view?usp=sharing)

### hyperparameters:
- `num_gradient_accumulation_steps`: meta-learning inner steps

## Zero-shot Transfer
- `UD_GT`: Same as [pre-training](#pre-training).
- `UD_ROOT`: Root path of treebank files used for testing. For scenarios that use ground truth text segmentation and POS tags as inputs to the parser, simply set it the same as `UD_GT`. For those who would like to use their own preprocessing pipeline (e.g. for comparison with [paper](https://www.aclweb.org/anthology/K18-2016.pdf)) to generate segmentation and POS tags as inputs to the parser, put all preprocessed conllu files in a singler folder and set `UD_ROOT` to it.
- `EPOCH_NUM`:  Which pre-training epoch checkpoint to perform zero-shot transfer from.
- `ZS_LANG`:  Language code of target transfer language (e.g. wo, te, cop, ..., etc.).
- `SUFFIX`:  Suffix of folder names storing results.
- `<serialization_dir>`:  Directory of model to perform zero-shot transfer from. For example, if one would like to perform zero-shot transfer from the pos-only multi-task baseline model, simply extract pre-trained model [multi-pos.tar.gz](https://drive.google.com/file/d/1v56inJzvpe9_YtJvtPUSaJeIIa2Fa2Qg/view?usp=sharing) and set `<serialization_dir>` to that folder.
```=bash
UD_GT="path/to/your/ud-treebanks-v2.x/**/" UD_ROOT="path/to/your/preprocessed/conllu-files/" bash zs-eval.sh <serialization_dir> $EPOCH_NUM $ZS_LANG 0 $SUFFIX
```

Results will be stored in log dir: `<serialization_dir>_${EPOCH_NUM}_${ZS_LANG}_${SUFFIX}`.

## Fine-tuning
- `UD_GT`: Same as [pre-training](#pre-training).
- `UD_ROOT`: Same as [zero-shot transfer](#zero-shot-transfer).
- `EPOCH_NUM`:  Which pre-training epoch checkpoint to perform fine-tuning from.
- `ZS_LANG`:  Code of target transfer language (e.g. wo, te, cop, ..., etc.).
- `NUM_EPOCHS`: Perform fine-tuning for this many number of epochs.
- `SUFFIX`:  Suffix of folder names storing results.
- `<serialization_dir>`: Directory of model to perform fine-tuning from. For example, if one would like to perform fine-tuning from the pos-only multi-task baseline model, simply extract pre-trained model [multi-pos.tar.gz](https://drive.google.com/file/d/1v56inJzvpe9_YtJvtPUSaJeIIa2Fa2Qg/view?usp=sharing) and set `<serialization_dir>` to that folder.
```=bash
UD_GT="path/to/your/ud-treebanks-v2.x/**/" UD_ROOT="path/to/your/preprocessed/testset/" bash fine-tune.sh <serialization_dir> $EPOCH_NUM $FT_LANG $NUM_EPOCHS $SUFFIX
``` 

Results will be stored in log dir: `<serialization_dir>_${EPOCH_NUM}_${FT_LANG}_${SUFFIX}`.

## Files in log directory
- `train-result.conllu`: System prediction of training set (`$UD_GT/$ZS_LANG*-train.conllu`).
- `dev-result.conllu`: System prediction of development set (`$UD_GT/$ZS_LANG*-dev.conllu`).
- `result.conllu`: System prediction of testing set (`$UD_ROOT/$ZS_LANG*-test.conllu`).
- `result-gt.conllu`: System prediction of testing set  (`$UD_GT/$ZS_LANG*-test.conllu`).
- `result.txt`: Performance (LAS, UAS, etc.) of `result.conllu` computed by `utils/conll18_ud_eval.py`, which is provided by [CoNLL 2018 Shared Task](http://universaldependencies.org/conll18/evaluation.html).
- `result-gt.txt`: Performance (LAS, UAS, etc.) of `result-gt.conllu` computed by `utils/error_analysis.py`, which is modified from [CoNLL 2018 Shared Task](http://universaldependencies.org/conll18/evaluation.html). Scores grouped by sentence length (`LASlen[sentence length lower bound][sentence length upper bound]`) and dependency length(`LASdep[dependency length]`) are added.
