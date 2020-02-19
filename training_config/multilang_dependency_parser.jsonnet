// Configuration for the multi-lingual dependency parser model based on:
// Schuster et al. "Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot Dependency Parsing"
// https://www.aclweb.org/anthology/papers/N/N19/N19-1162 (NAACL 2019)
//
// To recompute alignemts for ELMo, refer to: https://github.com/TalSchuster/CrossLingualELMo
// For the dataset, refer to https://github.com/ryanmcd/uni-dep-tb
local MAX_LEN = 512;
local BASE_READER(x, alternate=true) = {
    "type": "ud_multilang",
    "languages": [x],
    "alternate": alternate,
    "instances_per_file": 32,
    "is_first_pass_for_vocab": false,
    "lazy": true,
    "token_indexers": {
        "roberta": {
            "type": "transformer_pretrained_mismatched",
            "model_name": "xlm-roberta-base",
            "max_length": MAX_LEN,
        }
    },
    "use_language_specific_pos": false
};

local LANGS = ["et", "fi", "de", "en", "hi", "ja", "fr", "it",
   "la", "bg", "sl", "eu", "zh"];

local READERS(xs, alternate=true) = {
    [x]: BASE_READER(x, alternate) for x in xs
};

local UD_ROOT = "/home/nlpmaster/ssd-1t/corpus/ud/ud-treebanks-v2.5/UD_*/";
local DATA_PATH(lang, split) = UD_ROOT + lang + "*" + split + ".conllu";

local READERS(xs, alternate=true) = {
    [x]: BASE_READER(x, alternate) for x in xs
};

local UD_ROOT = "/home/nlpmaster/ssd-1t/corpus/ud/ud-treebanks-v2.5/UD_*/";
local DATA_PATH(lang, split) = UD_ROOT + lang + "*" + split +".conllu";

{
    "dataset_readers": READERS(LANGS),
    "validation_dataset_readers": READERS(LANGS, false),
    "vocabulary": {
        "type": "from_files",
        "directory": "data/vocabulary"
    },
    "iterator": {
        "type": "bucket",
        "batch_size": 8,
        "sorting_keys": [["words", "roberta___mask"]],
        "instances_per_epoch": 200
    },
    "validation_iterator": {
        "type": "bucket",
        "sorting_keys": [["words", "roberta___mask"]],
        "batch_size": 32
    },
    "model": {
        "type": "biaffine_parser_multilang",
        "arc_representation_dim": 500,
        "dropout": 0.33,
        "encoder": {
            "type": "lstm",
            "bidirectional": true,
            "dropout": 0.33,
            "hidden_size": 200,
            "input_size": 818,
            "num_layers": 2
        },
        "langs_for_early_stop": LANGS,
        "pos_tag_embedding": {
            "embedding_dim": 50,
            "vocab_namespace": "pos"
        },
        "tag_representation_dim": 100,
        "text_field_embedder": {
            "token_embedders": {
                "roberta": {
                    "type": "transformer_pretrained_mismatched",
                    "model_name": "xlm-roberta-base",
                    "requires_grad": false,
                    "max_length": MAX_LEN,
                }
            }
        }
    },
    // UDTB v2.0 is available at https://github.com/ryanmcd/uni-dep-tb
    // Set TRAIN_PATHNAME='std/**/*train.conll'
    "train_data_paths": {
        [lang]: DATA_PATH(lang, "train") for lang in LANGS
    },
    "validation_data_paths": {
        [lang]: DATA_PATH(lang, "dev") for lang in LANGS
    },
    "test_data_paths": {
        [lang]: DATA_PATH(lang, "test") for lang in LANGS
    },
    "trainer": {
        "type": "meta",
        "cuda_device": 0,
        "num_epochs": 40,
        "optimizer": {
          "type": "adam",
          "lr": 3e-4,
        },
        "patience": 10,
        "grad_norm": 5.0,
        "validation_metric": "+LAS_AVG",
        "num_gradient_accumulation_steps": 2,
        "wrapper": {
            "type": "reptile",
            "grad_norm": 5.0,
            "optimizer_cls": "Adam",
            "optimizer_kwargs": {
                "lr": 3e-4
            }
        },
        "wandb": {
            "name": "test-1",
            "project": "allennlp-maml-parsing",
            "tags": LANGS,
        },
    }
}
