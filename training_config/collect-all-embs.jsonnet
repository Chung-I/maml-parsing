// Configuration for the multi-lingual dependency parser model based on:
// Schuster et al. "Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot Dependency Parsing"
// https://www.aclweb.org/anthology/papers/N/N19/N19-1162 (NAACL 2019)
//
// To recompute alignemts for ELMo, refer to: https://github.com/TalSchuster/CrossLingualELMo
// For the dataset, refer to https://github.com/ryanmcd/uni-dep-tb
local NUM_EPOCHS = 1;
local MAX_LEN = 512;
local TOKEN_EMBEDDER_KEY = std.extVar("LM");
local MODEL_NAME = if TOKEN_EMBEDDER_KEY == "bert" then "bert-base-multilingual-cased" else "xlm-roberta-base";
local HIDDEN_SIZE = 128;
local BIDIR = true;
local NUM_DIRS = if BIDIR then 2 else 1;
local BS = 4;
local TAG_DIM = 256;
local BASE_READER(x, alternate=true) = {
    "type": "ud_multilang",
    "languages": [x],
    "alternate": alternate,
    "instances_per_file": 32,
    "is_first_pass_for_vocab": false,
    "lazy": false,
    "token_indexers": {
        [TOKEN_EMBEDDER_KEY]: {
            "type": "transformer_pretrained_mismatched",
            "model_name": MODEL_NAME,
            "max_length": MAX_LEN,
        }
    },
    "read_language": false,
    "use_language_specific_pos": false,
    "deprel_file": "data/vocabulary/head_tags.txt",
};

local LANG = std.extVar("FT_LANG");

local READER(x, alternate=true) = BASE_READER(x, alternate);

local UD_ROOT = std.extVar("UD_ROOT");
local DATA_PATH(lang, split) = UD_ROOT + lang + "*-ud-" + split + ".conllu";

{
    "dataset_reader": READER(LANG, false),
    "validation_dataset_reader": READER(LANG, false),
    "vocabulary": {
        "type": "from_files",
        "directory": "data/vocabulary"
    },
    "iterator": {
        "type": "bucket",
        "batch_size": BS,
        "sorting_keys": [["words", TOKEN_EMBEDDER_KEY + "___mask"]],
        #"maximum_samples_per_batch": [TOKEN_EMBEDDER_KEY + "___mask", BS * MAX_LEN],
    },
    "validation_iterator": {
        "type": "bucket",
        "batch_size": BS,
        "sorting_keys": [["words", TOKEN_EMBEDDER_KEY + "___mask"]],
        #"maximum_samples_per_batch": [TOKEN_EMBEDDER_KEY + "___mask", BS * MAX_LEN],
    },
    "model": {
        "type": "collect-all-embs",
        "arc_representation_dim": 100,
        "dropout": 0.33,
        "lexical_dropout": 0.33,
        "pos_dropout": 0.33,
        "input_dropout": 0.33,
        "tag_dim": TAG_DIM,
        "encoder": {
            "type": "lstm",
            "hidden_size": HIDDEN_SIZE,
            "input_size": 256,
            "num_layers": 3,
            "dropout": 0.0,
            "bidirectional": BIDIR,
        },
        "pos_tag_embedding": {
            "embedding_dim": 100,
            "vocab_namespace": "pos"
        },
        "vib": {
          "activation": "leaky_relu",
          "tag_dim": TAG_DIM,
          "type_token_reg": false,
          "sample_size": 5,
          "beta": 0.0,
        },
        "per_lang_vib": false,
        "langs_for_early_stop": [LANG],
        "tag_representation_dim": 50,
        "model_name": MODEL_NAME,
        "text_field_embedder": {
            "token_embedders": {
                [TOKEN_EMBEDDER_KEY]: {
                    "type": "transformer_pretrained_mismatched",
                    "model_name": MODEL_NAME,
                    "requires_grad": false,
                    "max_length": MAX_LEN,
                    "layer_dropout": 0.0,
                    "dropout": 0.0,
                    "combine_layers": std.extVar("LAYER_NUM"),
                }
            }
        },
    },
    // UDTB v2.0 is available at https://github.com/ryanmcd/uni-dep-tb
    // Set TRAIN_PATHNAME='std/**/*train.conll'
    "train_data_path": DATA_PATH(LANG, "train"),
    "validation_data_path": DATA_PATH(LANG, "dev"),
    "trainer": {
        "type": "collect-all-embs",
        "cuda_device": 0,
        "num_epochs": NUM_EPOCHS,
        "optimizer": {
          "type": "adam",
          "lr": 5e-5,
        },
        "learning_rate_scheduler": {
          "type": "slanted_triangular",
          "num_epochs": NUM_EPOCHS,
          "num_steps_per_epoch": 1000, // dummy value, modified in the code
        },
        "patience": 10,
        "grad_norm": 5.0,
        "validation_metric": "+LAS_AVG",
        "num_serialized_models_to_keep": 1,
        "num_gradient_accumulation_steps": 1,
        // "wandb": {
        //     "name": std.extVar("RUN_NAME"),
        //     "project": "allennlp-maml-parsing",
        // },
    }
}
