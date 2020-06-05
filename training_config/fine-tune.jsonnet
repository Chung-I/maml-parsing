// Configuration for the multi-lingual dependency parser model based on:
// Schuster et al. "Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot Dependency Parsing"
// https://www.aclweb.org/anthology/papers/N/N19/N19-1162 (NAACL 2019)
//
// To recompute alignemts for ELMo, refer to: https://github.com/TalSchuster/CrossLingualELMo
// For the dataset, refer to https://github.com/ryanmcd/uni-dep-tb
local MAX_LEN = 512;
local MODEL_NAME = "bert-base-multilingual-cased";
local NUM_EPOCHS = std.parseInt(std.extVar("NUM_EPOCHS"));
local BS = 8;
local TOKEN_EMBEDDER_KEY = "bert";
local BASE_READER(x, alternate=true) = {
    "type": "ud_multilang",
    "languages": [x],
    "alternate": alternate,
    "instances_per_file": 32,
    "is_first_pass_for_vocab": false,
    "lazy": false,
    "max_len": 256,
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

local CV = std.extVar("CV");

local UD_ROOT = std.extVar("UD_ROOT");
local DATA_PATH(lang, split) = UD_ROOT + lang + CV + "**-" + split + ".conllu";

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
        "maximum_samples_per_batch": [TOKEN_EMBEDDER_KEY + "___mask", BS * MAX_LEN],
    },
    "validation_iterator": {
        "type": "bucket",
        "batch_size": BS,
        "sorting_keys": [["words", TOKEN_EMBEDDER_KEY + "___mask"]],
        "maximum_samples_per_batch": [TOKEN_EMBEDDER_KEY + "___mask", BS * MAX_LEN],
    },
    "model": {
      "type": "from_archive",
      "archive_file": std.extVar("ARCHIVE_PATH"),
    },
    // UDTB v2.0 is available at https://github.com/ryanmcd/uni-dep-tb
    // Set TRAIN_PATHNAME='std/**/*train.conll'
    "train_data_path": DATA_PATH(LANG, "train"),
    "validation_data_path": DATA_PATH(LANG, "dev"),
    "trainer": {
        "type": "wandb",
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
        "ft_lang_mean_dir": "ckpts/" + LANG + "_mean_fixed-bert",
        "validation_metric": "+LAS_AVG",
        "num_serialized_models_to_keep": 1,
        "num_gradient_accumulation_steps": 1,
        // "wandb": {
        //     "name": std.extVar("RUN_NAME"),
        //     "project": "allennlp-maml-parsing",
        // },
    }
}
