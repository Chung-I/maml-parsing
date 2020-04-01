// Configuration for the multi-lingual dependency parser model based on:
// Schuster et al. "Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot Dependency Parsing"
// https://www.aclweb.org/anthology/papers/N/N19/N19-1162 (NAACL 2019)
//
// To recompute alignemts for ELMo, refer to: https://github.com/TalSchuster/CrossLingualELMo
// For the dataset, refer to https://github.com/ryanmcd/uni-dep-tb
local MAX_LEN = 512;
local MODEL_NAME = "xlm-roberta-base";
local NUM_EPOCHS = 10;
local HIDDEN_SIZE = 400;
local ADV_HIDDEN = 128;
local BIDIR = true;
local NUM_DIRS = if BIDIR then 2 else 1;

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

local TRAIN_LANGS = ['af', 'grc', 'pt', 'sv', 'no', 'es', 'zh', 'fro', 'ja', 'tr', 'hi', 'ar', 'ca', 'hr', 'el', 'hu', 'la', 'fr', 'fi', 'eu', 'ko', 'et', 'id', 'fa', 'uk', 'got', 'pl', 'ug', 'vi', 'da', 'ru', 'gl', 'it', 'cu', 'cs', 'he', 'sr', 'en', 'sk', 'bg', 'sl', 'ur', 'nl', 'lv', 'de', 'ro'];

local READERS(xs, alternate=true) = {
    [x]: BASE_READER(x, alternate) for x in xs
};

local UD_ROOT = std.extVar("UD_ROOT");
local DATA_PATH(lang, split) = UD_ROOT + lang + "*-ud-" + split + ".conllu";

{
    "dataset_readers": READERS(TRAIN_LANGS),
    "vocabulary": {
        "type": "from_files",
        "directory": "data/vocabulary"
    },
    "iterator": {
        "type": "bucket",
        "batch_size": 16,
        "sorting_keys": [["words", "roberta___mask"]],
        "instances_per_epoch": 160000,
    },
    "model": {
        "type": "ud_biaffine_parser_multilang",
        "arc_representation_dim": 500,
        "dropout": 0.33,
        "word_dropout": 0.33,
        "input_dropout": 0.33,
        "encoder": {
            "type": "lstm",
            "hidden_size": HIDDEN_SIZE,
            "input_size": 868,
            "num_layers": 3,
            "dropout": 0.0,
            "bidirectional": BIDIR,
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
                    "model_name": "xlm-roberta-base",
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
    "trainer": {
        "type": "meta",
        "cuda_device": 0,
        "num_epochs": NUM_EPOCHS,
        "optimizer": {
          "type": "adam",
          "lr": 3e-4,
        },
        "patience": 10,
        "validation_metric": "+LAS_AVG",
        "save_embedder": false,
        "num_serialized_models_to_keep": -1,
        "num_gradient_accumulation_steps": 1,
        "tasks_per_step": 10,
        "wrapper": {
            "type": "fomaml",
            "optimizer_cls": "Adam",
            "optimizer_kwargs": {
                "lr": 3e-4
            }
        },
        "task_discriminator": {
          "encoder": {
            "type": "pool-cnn",
            "embedding_dim": HIDDEN_SIZE * NUM_DIRS,
            "num_filters": 128,
          },
          "first_n_states": 4,
          "steps_per_update": 1,
        },
        "discriminator_optimizer": {
          "type": "sgd",
          "lr": 0.01,
          "momentum": 0.9,
          "nesterov": true,
        },
    }
}
