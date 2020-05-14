// Configuration for the multi-lingual dependency parser model based on:
// Schuster et al. "Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot Dependency Parsing"
// https://www.aclweb.org/anthology/papers/N/N19/N19-1162 (NAACL 2019)
//
// To recompute alignemts for ELMo, refer to: https://github.com/TalSchuster/CrossLingualELMo
// For the dataset, refer to https://github.com/ryanmcd/uni-dep-tb
local MAX_LEN = 512;
local MODEL_NAME = "xlm-roberta-base";
local LM_DIM = 768;
local NUM_EPOCHS = 10;
local HIDDEN_SIZE = 128;
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
    "use_language_specific_pos": false,
    "use_language_specific_deprel": false,
    // "in_memory": true,
};

local TRAIN_LANGS = ['af', 'grc', 'pt', 'sv', 'no', 'es', 'zh', 'fro', 'ja', 'tr', 'hi', 'ar', 'ca', 'hr', 'el', 'hu', 'la', 'fr', 'fi', 'eu', 'ko', 'et', 'id', 'fa', 'uk', 'got', 'pl', 'ug', 'vi', 'da', 'ru', 'gl', 'it', 'cu', 'cs', 'he', 'sr', 'en', 'sk', 'bg', 'sl', 'ur', 'nl', 'lv', 'de', 'ro'];
// local TRAIN_LANGS = ['ar', 'he', 'gl', 'pt', 'it', 'id', 'fr', 'cu', 'es', 'ca', 'ro', 'vi', 'bg', 'el', 'got', 'cs', 'pl', 'la', 'en', 'sv', 'uk', 'no', 'hr'];

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
        "instances_per_epoch": 80000,
    },
    "model": {
        "type": "ud_biaffine_parser_multilang_vib",
        "arc_representation_dim": 100,
        "dropout": 0.33,
        "lexical_dropout": 0.33,
        "pos_dropout": 0.33,
        "dropout_location": "lm",
        "input_dropout": 0.33,
        "tag_dim": LM_DIM,
        "encoder": {
            "type": "pass_through",
            "input_dim": LM_DIM,
        },
        "per_lang_vib": false,
        "lang_mean_regex": "ckpts/*_mean",
        "langs_for_early_stop": TRAIN_LANGS,
        "tag_representation_dim": 50,
        "model_name": MODEL_NAME,
        "text_field_embedder": {
            "token_embedders": {
                "roberta": {
                    "type": "transformer_pretrained_mismatched",
                    "model_name": "xlm-roberta-base",
                    "requires_grad": true,
                    "max_length": MAX_LEN,
                    "layer_dropout": 0.0,
                    "dropout": 0.0,
                    "combine_layers": "last",
                }
            }
        },
        "initializer": [
          // [".*VIB.*encoder.*weight", {"type": "xavier_uniform"}],
          // [".*VIB.*encoder.*bias", {"type": "zero"}],
          // [".*r_mean", {"type": "xavier_uniform"}],
          // [".*r_std", {"type": "uniform"}],
          [".*projection.*weight", {"type": "xavier_uniform"}],
          [".*projection.*bias", {"type": "zero"}],
          [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
          [".*tag_bilinear.*bias", {"type": "zero"}],
          [".*weight_ih.*", {"type": "xavier_uniform"}],
          [".*weight_hh.*", {"type": "orthogonal"}],
          [".*bias_ih.*", {"type": "zero"}],
          [".*bias_hh.*", {"type": "lstm_hidden_bias"}]]
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
          "lr": 5e-5,
        },
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 80,
            "num_steps_per_epoch": 1250,
        },
        "patience": 10,
        "validation_metric": "+LAS_AVG",
        "save_embedder": true,
        "num_serialized_models_to_keep": -1,
        "num_gradient_accumulation_steps": 2,
        "tasks_per_step": 10,
        "wrapper": {
            "type": "maml",
            "optimizer_cls": "Adam",
            "optimizer_kwargs": {
                "lr": 5e-5,
            },
        },
    }
}
