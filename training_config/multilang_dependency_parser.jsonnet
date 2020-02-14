// Configuration for the multi-lingual dependency parser model based on:
// Schuster et al. "Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot Dependency Parsing"
// https://www.aclweb.org/anthology/papers/N/N19/N19-1162 (NAACL 2019)
//
// To recompute alignemts for ELMo, refer to: https://github.com/TalSchuster/CrossLingualELMo
// For the dataset, refer to https://github.com/ryanmcd/uni-dep-tb
local BASE_READER(x) = {
    "type": "ud_multilang",
    "languages": [x],
    "alternate": true,
    "instances_per_file": 32,
    "is_first_pass_for_vocab": true,
    "lazy": true,
    "token_indexers": {
        "roberta": {
            "type": "pretrained_transformer_mismatched",
            "model_name": "xlm-roberta-base"
        }
    },
    "use_language_specific_pos": false
};

local UD_ROOT = "/home/nlpmaster/ssd-1t/corpus/ud/ud-treebanks-v2.5/UD_";

{
    "dataset_reader": {
        "type": "meta",
        "readers": {
            "en": BASE_READER("en"),
            "zh": BASE_READER("zh"),
            "fr": BASE_READER("fr")
        }
    },
    "iterator": {
        "type": "same_language",
        "batch_size": 32,
        "sorting_keys": [["words", "num_tokens"]],
        "instances_per_epoch": 32000
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
            "input_size": 1074,
            "num_layers": 3
        },
        "langs_for_early_stop": [
            "en",
            "de",
            "it",
            "fr",
            "pt",
            "sv"
        ],
        "pos_tag_embedding": {
            "embedding_dim": 50,
            "vocab_namespace": "pos"
        },
        "tag_representation_dim": 100,
        "text_field_embedder": {
            "token_embedders": {
                "roberta": {
                    "type": "pretrained_transformer_mismatched",
                    "model_name": "xlm-roberta-base"
                }
            }
        }
    },
    // UDTB v2.0 is available at https://github.com/ryanmcd/uni-dep-tb
    // Set TRAIN_PATHNAME='std/**/*train.conll'
    "train_data_path": {
            "en": UD_ROOT + "English-*/*-train.conllu",
            "zh": UD_ROOT + "Chinese-*/*-train.conllu",
            "fr": UD_ROOT + "French-*/*-train.conllu"
    },
    "validation_data_path": {
            "en": UD_ROOT + "English-*/*-dev.conllu",
            "zh": UD_ROOT + "Chinese-*/*-dev.conllu",
            "fr": UD_ROOT + "French-*/*-dev.conllu"
    },
    "test_data_path": {
            "en": UD_ROOT + "English-*/*-test.conllu",
            "zh": UD_ROOT + "Chinese-*/*-test.conllu",
            "fr": UD_ROOT + "French-*/*-test.conllu"
    },
    "trainer": {
        "type": "meta",
        "cuda_device": 0,
        "num_epochs": 40,
        "optimizer": "adam",
        "patience": 10,
        "validation_metric": "+LAS_AVG"
    }
}
