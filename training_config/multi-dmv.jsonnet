// Configuration for the multi-lingual dependency parser model based on:
// Schuster et al. "Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot Dependency Parsing"
// https://www.aclweb.org/anthology/papers/N/N19/N19-1162 (NAACL 2019)
//
// To recompute alignemts for ELMo, refer to: https://github.com/TalSchuster/CrossLingualELMo
// For the dataset, refer to https://github.com/ryanmcd/uni-dep-tb
local MAX_LEN = 512;
local MODEL_NAME = "xlm-roberta-base";
local INPUT_SIZE = 768;
local NUM_EPOCHS = 10;
local HIDDEN_SIZE = 128;
local BIDIR = true;
local NUM_DIRS = if BIDIR then 2 else 1;
local TAG_DIM = 256;

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
    "max_len": 15,
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
        "batch_size": 32,
        "sorting_keys": [["words", "roberta___mask"]],
        "instances_per_epoch": 32000,
    },
    "model": {
        "type": "neuraldmv",
        "dmv": {
            "cvalency_num": 1,
            "valency_num": 2,
        },
        "nice_layer": {
            "couple_layers": 4,
            "cell_layers": 1,
            "feat_dim": INPUT_SIZE,
            "hidden_units": 64,
        },
        "encoder": {
            "type": "lstm",
            "hidden_size": 32,
            "input_size": INPUT_SIZE,
            "num_layers": 1,
            "dropout": 0.0,
            "bidirectional": true,
        },
        "text_field_embedder": {
            "token_embedders": {
                "roberta": {
                    "type": "transformer_pretrained_mismatched",
                    "model_name": "xlm-roberta-base",
                    "requires_grad": false,
                    "max_length": MAX_LEN,
                    "layer_dropout": 0.0,
                    "dropout": 0.0,
                    "combine_layers": "last",
                }
            }
        },
        "hidden_dim": 32,
        "state_dim": 32,
        "n_states": 30,
        "lang_mean_regex": "ckpts/*_mean",
        "initializer": [
            ["state_emb", {"type": "xavier_uniform"}],
            ["r_mean", {"type": "xavier_uniform"}],
            ["r_std", {"type": "uniform"}],
            //[".*_mlp", {"type": "xavier_uniform"}],
            //["stop_.*", {"type": "xavier_uniform"}],
        ],
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
        "grad_norm": 1.0,
        "patience": 10,
        "validation_metric": "+LAS_AVG",
        "save_embedder": false,
        "num_serialized_models_to_keep": -1,
        "num_gradient_accumulation_steps": 1,
        "tasks_per_step": 10,
        "wrapper": {
            "type": "multi",
        },
    }
}
