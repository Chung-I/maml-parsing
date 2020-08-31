// Configuration for the multi-lingual dependency parser model based on:
// Schuster et al. "Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot Dependency Parsing"
// https://www.aclweb.org/anthology/papers/N/N19/N19-1162 (NAACL 2019)
//
// To recompute alignemts for ELMo, refer to: https://github.com/TalSchuster/CrossLingualELMo
// For the dataset, refer to https://github.com/ryanmcd/uni-dep-tb
local MAX_LEN = 512;
local MODEL_NAME = "bert-base-multilingual-cased";
local NUM_EPOCHS = 10;
local HIDDEN_SIZE = 100;
local BIDIR = true;
local NUM_DIRS = if BIDIR then 2 else 1;
local TAG_DIM = 100;
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
local TRAIN_LANGS = ['af', 'grc', 'pt', 'sv', 'no', 'es', 'zh', 'fro', 'ja', 'tr', 'hi', 'ar', 'ca', 'hr', 'el', 'hu', 'la', 'fr', 'fi', 'eu', 'ko', 'et', 'id', 'fa', 'uk', 'got', 'pl', 'ug', 'vi', 'da', 'ru', 'gl', 'it', 'cu', 'cs', 'he', 'sr', 'en', 'sk', 'bg', 'sl', 'ur', 'nl', 'lv', 'de', 'ro'];

local READERS(xs, alternate=true) = {
    [x]: BASE_READER(x, alternate) for x in xs
};

local UD_ROOT = std.extVar("UD_ROOT");
local DATA_PATH(lang, split) = UD_ROOT + lang + "*-ud-" + split + ".conllu";

{
    "dataset_readers": READERS(TRAIN_LANGS),
    // "random_seed": 531,
    // "numpy_seed": 531,
    "vocabulary": {
        "type": "from_files",
        "directory": "data/vocabulary"
    },
    "iterator": {
        "type": "bucket",
        "batch_size": 16,
        "sorting_keys": [["words", TOKEN_EMBEDDER_KEY + "___mask"]],
        "instances_per_epoch": 64000,
    },
    "model": {
        "type": "ud_biaffine_parser_multilang_vib",
        "arc_representation_dim": 200,
        "tag_representation_dim": 200,
        "dropout": 0.0,
        "token_embedder_key": TOKEN_EMBEDDER_KEY,
        "pos_tag_embedding": {
            "embedding_dim": 100,
            "vocab_namespace": "pos"
        },
        "lexical_dropout": 1.0,
        "pos_dropout": 0.33,
        "dropout_location": "input",
        "input_dropout": 0.0,
        "tag_dim": TAG_DIM,
        "encoder": {
            "type": "lstm",
            "hidden_size": HIDDEN_SIZE,
            "input_size": 100,
            "num_layers": 3,
            "dropout": 0.0,
            "bidirectional": BIDIR,
        },
        "per_lang_vib": false,
        "langs_for_early_stop": TRAIN_LANGS,
        "model_name": MODEL_NAME,
        "initializer": [
          [".*projection.*weight", {"type": "xavier_uniform"}],
          [".*projection.*bias", {"type": "zero"}],
          [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
          [".*tag_bilinear.*bias", {"type": "zero"}],
          [".*weight_ih.*", {"type": "xavier_uniform"}],
          [".*weight_hh.*", {"type": "orthogonal"}],
          [".*bias_ih.*", {"type": "zero"}],
          [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
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
            "type": "adamw",
            "weight_decay": 0.01,
            "lr": 3e-4,
            "betas": [0.9, 0.9],
        },
        "learning_rate_scheduler": {
            "type": "multi_step",
            "milestones": [2, 6, 10, 14, 20],
            "gamma": 0.5,
        },
        "grad_clipping": 5.0,
        "patience": 10,
        "validation_metric": "+LAS_AVG",
        "save_embedder": true,
        "num_serialized_models_to_keep": 1,
        "num_gradient_accumulation_steps": 2,
        "tasks_per_step": 10,
        "wrapper": {
            "type": "maml",
            "optimizer_cls": "Adam",
            "optimizer_kwargs": {
                "lr": 3e-4,
            },
        },
    }
}
