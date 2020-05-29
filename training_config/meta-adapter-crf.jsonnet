// Configuration for the multi-lingual dependency parser model based on:
// Schuster et al. "Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot Dependency Parsing"
// https://www.aclweb.org/anthology/papers/N/N19/N19-1162 (NAACL 2019)
//
// To recompute alignemts for ELMo, refer to: https://github.com/TalSchuster/CrossLingualELMo
// For the dataset, refer to https://github.com/ryanmcd/uni-dep-tb
local MAX_LEN = 512;
local MODEL_NAME = "bert-base-multilingual-cased";
local NUM_EPOCHS = 20;
local BS = 16;
local LM_DIM = 768;
local HIDDEN_SIZE = 128;
local BIDIR = true;
local NUM_DIRS = if BIDIR then 2 else 1;
local TAG_DIM = 768;
local TOKEN_EMBEDDER_KEY = "bert";

local BASE_READER(x, alternate=true) = {
    "type": "ud_multilang",
    "languages": [x],
    "alternate": alternate,
    "instances_per_file": 1600,
    "is_first_pass_for_vocab": false,
    "lazy": true,
    "max_len": 256,
    "token_indexers": {
        [TOKEN_EMBEDDER_KEY]: {
            "type": "transformer_pretrained_mismatched",
            "model_name": MODEL_NAME,
            "max_length": MAX_LEN,
        }
    },
    "use_language_specific_pos": false,
    "use_language_specific_deprel": false,
    "read_language": false,
};

// local TRAIN_LANGS = ['ar_padt', 'eu_bdt', 'zh_gsd', 'en_ewt', 'fi_tdt', 'he_htb', 'hi_hdtb', 'it_isdt', 'ja_gsd', 'ko_gsd', 'ru_syntagrus', 'sv_talbanken', 'tr_imst'];
// local TRAIN_LANGS = ['bg', 'hr', 'cs', 'pl', 'ru', 'sk', 'sl', 'uk'];
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
        "maximum_samples_per_batch": [TOKEN_EMBEDDER_KEY + "___mask", BS * 256],
        "instances_per_epoch": 64000,
    },
    "model": {
        "text_field_embedder": {
            "token_embedders": {
                [TOKEN_EMBEDDER_KEY]: {
                    "type": "transformer_pretrained_mismatched",
                    "model_name": "adapter_" + MODEL_NAME,
                    "requires_grad": false,
                    "adapter_size": 256,
                    "max_length": MAX_LEN,
                    "layer_dropout": 0.1,
                    "bert_dropout": 0.5,
                    "dropout": 0.2,
                    "combine_layers": "mix",
                }
            }
        },
        "type": "ud_biaffine_parser_multilang_vib",
        "arc_representation_dim": 768,
        "use_crf": true,
        "dropout": 0.0,
        "token_embedder_key": TOKEN_EMBEDDER_KEY,
        "lexical_dropout": 0.2,
        "pos_dropout": 0.0,
        "dropout_location": "input",
        "input_dropout": 0.0,
        "tag_dim": TAG_DIM,
        "encoder": {
            "type": "pass_through",
            "input_dim": 768
        },
        "per_lang_vib": false,
        "langs_for_early_stop": TRAIN_LANGS,
        "tag_representation_dim": 256,
        "model_name": MODEL_NAME,
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
            "type": "adamw",
            "weight_decay": 0.01,
            "lr": 1e-3,
            "betas": [0.9, 0.99],
        },
        "learning_rate_scheduler": {
            "type": "ulmfit_sqrt",
            "model_size": 1,
            "warmup_steps": 1231,
            "start_step": 0,
            "factor": 35.0,
            "decay_factor": 0.4,
        },
        "grad_norm": 10.0,
        "patience": 10,
        "validation_metric": "+LAS_AVG",
        "save_embedder": false,
        "num_serialized_models_to_keep": -1,
        "num_gradient_accumulation_steps": 2,
        "tasks_per_step": 10,
        "wrapper": {
            "type": "reptile",
            "optimizer_cls": "Adam",
            "optimizer_kwargs": {
                "lr": 3e-4
            },
            // "inherit": true,
        },
    }
}
