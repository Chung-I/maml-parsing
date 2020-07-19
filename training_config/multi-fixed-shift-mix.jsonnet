// Configuration for the multi-lingual dependency parser model based on:
// Schuster et al. "Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot Dependency Parsing"
// https://www.aclweb.org/anthology/papers/N/N19/N19-1162 (NAACL 2019)
//
// To recompute alignemts for ELMo, refer to: https://github.com/TalSchuster/CrossLingualELMo
// For the dataset, refer to https://github.com/ryanmcd/uni-dep-tb
local MAX_LEN = 512;
local MODEL_NAME = "bert-base-multilingual-cased";
local NUM_EPOCHS = 10;
local HIDDEN_SIZE = 384;
local BIDIR = true;
local BS = 16;
local NUM_DIRS = if BIDIR then 2 else 1;
local TAG_DIM = 768;
local TOKEN_EMBEDDER_KEY = "bert";

local BASE_READER(x, alternate=true) = {
    "type": "ud_multilang",
    "languages": [x],
    "alternate": alternate,
    "instances_per_file": 32,
    "is_first_pass_for_vocab": false,
    "lazy": true,
    "token_indexers": {
        [TOKEN_EMBEDDER_KEY]: {
            "type": "transformer_pretrained_mismatched",
            "model_name": MODEL_NAME,
            "max_length": MAX_LEN,
        }
    },
    "use_language_specific_pos": false,
    "use_language_specific_deprel": false,
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
        "batch_size": BS,
        "sorting_keys": [["words", TOKEN_EMBEDDER_KEY + "___mask"]],
        // "maximum_samples_per_batch": [TOKEN_EMBEDDER_KEY + "___mask", BS * 256],
        "instances_per_epoch": 160000,
    },
    "model": {
        "type": "ud_biaffine_parser_multilang_vib",
        "arc_representation_dim": 768,
        "dropout": 0.2,
        "lexical_dropout": 0.2,
        "pos_dropout": 0.2,
        "dropout_location": "lm",
        "input_dropout": 0.0,
        "tag_dim": TAG_DIM,
        "use_crf": false,
        // "encoder": {
        //     "type": "pass_through",
        //     "input_dim": 768
        // },
        "encoder": {
            "type": "lstm",
            "hidden_size": HIDDEN_SIZE,
            "input_size": 768,
            "num_layers": 1,
            "dropout": 0.0,
            "bidirectional": BIDIR,
        },
        // "vib": {
        //   "activation": "leaky_relu",
        //   "tag_dim": TAG_DIM,
        //   "type_token_reg": false,
        //   "sample_size": 5,
        //   "beta": 0.0,
        // },
        "per_lang_vib": false,
        // "lang_mean_regex": "ckpts/*_mean",
        "langs_for_early_stop": TRAIN_LANGS,
        "tag_representation_dim": 256,
        "model_name": MODEL_NAME,
        "text_field_embedder": {
            "token_embedders": {
                [TOKEN_EMBEDDER_KEY]: {
                    "type": "transformer_pretrained_mismatched",
                    "model_name": MODEL_NAME,
                    "requires_grad": false,
                    "max_length": MAX_LEN,
                    "layer_dropout": 0.1,
                    "bert_dropout": 0.5,
                    "dropout": 0.0,
                    "combine_layers": "mix",
                    "mean_affix": "all-mean-fixed-bert",
                    "lang_file": "data/vocabulary/lang_labels.txt"
                }
            }
        },
        "initializer": [
          [".*VIB.*encoder.*weight", {"type": "xavier_uniform"}],
          [".*VIB.*encoder.*bias", {"type": "zero"}],
          [".*r_mean", {"type": "xavier_uniform"}],
          [".*r_std", {"type": "uniform"}],
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
          "lr": 3e-4,
        },
        "patience": 10,
        "validation_metric": "+LAS_AVG",
        "save_embedder": true,
        "num_serialized_models_to_keep": -1,
        "num_gradient_accumulation_steps": 1,
        "tasks_per_step": 10,
        "wrapper": {
            "type": "multi",
        },
    }
}
