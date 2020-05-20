// Configuration for the multi-lingual dependency parser model based on:
// Schuster et al. "Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot Dependency Parsing"
// https://www.aclweb.org/anthology/papers/N/N19/N19-1162 (NAACL 2019)
//
// To recompute alignemts for ELMo, refer to: https://github.com/TalSchuster/CrossLingualELMo
// For the dataset, refer to https://github.com/ryanmcd/uni-dep-tb
local MAX_LEN = 512;
local MODEL_NAME = "bert-base-multilingual-cased";
local NUM_EPOCHS = 10;
local LM_DIM = 768;
local HIDDEN_SIZE = 128;
local BIDIR = true;
local NUM_DIRS = if BIDIR then 2 else 1;
local TAG_DIM = 768;

local BASE_READER(x, alternate=true) = {
    "type": "ud_multilang",
    "languages": [x],
    "alternate": alternate,
    "instances_per_file": 32,
    "is_first_pass_for_vocab": false,
    "lazy": true,
    "token_indexers": {
        "bert": {
            "type": "transformer_pretrained_mismatched",
            "model_name": MODEL_NAME,
            "max_length": MAX_LEN,
        }
    },
    "use_language_specific_pos": false,
    "use_language_specific_deprel": false,
    "read_language": false,
};

local TRAIN_LANGS = ['ar_padt', 'eu_bdt', 'zh_gsd', 'en_ewt', 'fi_tdt', 'he_htb', 'hi_hdtb',
'it_isdt', 'ja_gsd', 'ko_gsd', 'ru_syntagrus', 'sv_talbanken', 'tr_imst'];

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
        "sorting_keys": [["words", "bert___mask"]],
        "instances_per_epoch": 160000,
    },
    "model": {
        "text_field_embedder": {
            "token_embedders": {
                "bert": {
                    "type": "transformer_pretrained_mismatched",
                    "model_name": "adapter_" + MODEL_NAME,
                    "requires_grad": false,
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
        "dropout": 0.33,
        "lexical_dropout": 0.33,
        "pos_dropout": 0.33,
        "dropout_location": "lm",
        "input_dropout": 0.33,
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
            "parameter_groups": [
                [["^text_field_embedder.*.bert_model.embeddings",
                "^text_field_embedder.*.bert_model.encoder"], {}],
                [["^text_field_embedder.*._scalar_mix",
                "^text_field_embedder.*.pooler",
                "^scalar_mix",
                "^decoders",
                "^shared_encoder"], {}]
            ]
        },
        "learning_rate_scheduler": {
            "type": "ulmfit_sqrt",
            "model_size": 1,
            "warmup_steps": 8000,
            "start_step": 21695,
            "factor": 5.0,
            "decay_factor": 0.04,
        },
        "grad_norm": 5.0,
        "patience": 10,
        "validation_metric": "+LAS_AVG",
        "save_embedder": false,
        "num_serialized_models_to_keep": -1,
        "num_gradient_accumulation_steps": 1,
        "tasks_per_step": std.length(TRAIN_LANGS),
        "wrapper": {
            "type": "multi",
        },
    }
}
