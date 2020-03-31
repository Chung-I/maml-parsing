// Configuration for the multi-lingual dependency parser model based on:
// Schuster et al. "Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot Dependency Parsing"
// https://www.aclweb.org/anthology/papers/N/N19/N19-1162 (NAACL 2019)
//
// To recompute alignemts for ELMo, refer to: https://github.com/TalSchuster/CrossLingualELMo
// For the dataset, refer to https://github.com/ryanmcd/uni-dep-tb
local MAX_LEN = 512;
local MODEL_NAME = "xlm-roberta-base";
local NUM_EPOCHS = std.parseInt(std.extVar("NUM_EPOCHS"));
local WEIGHT_DROP = if std.extVar("WEIGHT_DROP") == "true" then 0.33 else 0.0;
local HIDDEN_SIZE = 400;
local BS = 16;
local INSTANCES_PER_EPOCH = 8000;
local STEPS_PER_EPOCH = INSTANCES_PER_EPOCH / BS;
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
            "model_name": MODEL_NAME,
            "max_length": MAX_LEN,
        }
    },
    "use_language_specific_pos": false
};

local LANG = std.extVar("FT_LANG");

local TRAIN_LANGS = [LANG];

local DEV_LANGS = TRAIN_LANGS;

local TEST_LANGS = [LANG];

local READERS(xs, alternate=true) = {
    [x]: BASE_READER(x, alternate) for x in xs
};

local UD_ROOT = std.extVar("UD_ROOT");
local DATA_PATH(lang, split) = UD_ROOT + lang + "*-ud-" + split + ".conllu";

{
    "dataset_readers": READERS(TRAIN_LANGS),
    "validation_dataset_readers": READERS(DEV_LANGS, false),
    "vocabulary": {
        "type": "from_files",
        "directory": "data/vocabulary"
    },
    "iterator": {
        "type": "bucket",
        "batch_size": BS,
        "sorting_keys": [["words", "roberta___mask"]],
        "instances_per_epoch": INSTANCES_PER_EPOCH,
    },
    "validation_iterator": {
        "type": "bucket",
        "sorting_keys": [["words", "roberta___mask"]],
        "batch_size": BS,
    },
    "model": {
        "type": "ud_biaffine_parser_multilang",
        "arc_representation_dim": 500,
        "dropout": 0.33,
        "word_dropout": 0.33,
        "input_dropout": 0.33,
        "encoder": {
            "type": "pass_through",
            "input_dim": 818,
        },
        "langs_for_early_stop": TRAIN_LANGS,
        "pos_tag_embedding": {
            "embedding_dim": 100,
            "vocab_namespace": "pos"
        },
        "tag_representation_dim": 100,
        "model_name": MODEL_NAME,
        "text_field_embedder": {
            "token_embedders": {
                "roberta": {
                    "type": "transformer_pretrained_mismatched",
                    "model_name": MODEL_NAME,
                    "requires_grad": false,
                    "max_length": MAX_LEN,
                    "layer_dropout": 0.1,
                    "dropout": 0.1,
                }
            }
        }
    },
    // UDTB v2.0 is available at https://github.com/ryanmcd/uni-dep-tb
    // Set TRAIN_PATHNAME='std/**/*train.conll'
    "train_data_paths": {
        [lang]: DATA_PATH(lang, "train") for lang in TRAIN_LANGS
    },
    "validation_data_paths": {
        [lang]: DATA_PATH(lang, "dev") for lang in DEV_LANGS
    },
    "test_data_paths": {
        [lang]: DATA_PATH(lang, "test") for lang in TEST_LANGS
    },
    "trainer": {
        "type": "meta",
        "cuda_device": -1,
        "num_epochs": NUM_EPOCHS,
        "optimizer": {
          "type": "adam",
          "lr": 3e-4,
        },
        "learning_rate_scheduler": {
          "type": "noam",
          "model_size": HIDDEN_SIZE,
          "warmup_steps": 1000,
        },
        "patience": 10,
        "validation_metric": "+LAS_AVG",
        "save_embedder": false,
        "num_serialized_models_to_keep": 1,
        "num_gradient_accumulation_steps": 1,
        "wrapper": {
            "type": "multi",
        },
        // "wandb": {
        //     "name": std.extVar("RUN_NAME") + "_" + LANG,
        //     "project": "allennlp-maml-parsing",
        // },
    }
}
