// Configuration for the multi-lingual dependency parser model based on:
// Schuster et al. "Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot Dependency Parsing"
// https://www.aclweb.org/anthology/papers/N/N19/N19-1162 (NAACL 2019)
//
// To recompute alignemts for ELMo, refer to: https://github.com/TalSchuster/CrossLingualELMo
// For the dataset, refer to https://github.com/ryanmcd/uni-dep-tb
local MAX_LEN = 512;
local MODEL_NAME = "xlm-roberta-base";
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

//local LANGS = ["et", "fi", "de", "en", "hi", "ja", "fr", "it",
//   "la", "bg", "sl", "eu", "zh"];

local TRAIN_LANGS = ['af_afribooms', 'grc_proiel', 'grc_perseus', 'ar_padt', 'hy_armtdp', 'eu_bdt', 'bg_btb', 'bxr_bdt', 'ca_ancora', 'zh_gsd', 'hr_set', 'cs_cac', 'cs_fictree', 'cs_pdt', 'da_ddt', 'nl_alpino', 'nl_lassysmall', 'en_ewt', 'en_gum', 'en_lines', 'et_edt', 'fi_ftb', 'fi_tdt', 'fr_gsd', 'fr_sequoia', 'fr_spoken', 'gl_ctg', 'gl_treegal', 'de_gsd', 'got_proiel', 'el_gdt', 'he_htb', 'hi_hdtb', 'hu_szeged', 'id_gsd', 'ga_idt', 'it_isdt', 'it_postwita', 'ja_gsd', 'kk_ktb', 'ko_gsd', 'ko_kaist', 'kmr_mg', 'la_ittb', 'la_proiel', 'la_perseus', 'lv_lvtb', 'sme_giella', 'no_bokmaal', 'no_nynorsk', 'no_nynorsklia', 'cu_proiel', 'fro_srcmf', 'fa_seraji', 'pl_lfg', 'pl_sz', 'pt_bosque', 'ro_rrt', 'ru_syntagrus', 'ru_taiga', 'sr_set', 'sk_snk', 'sl_ssj', 'sl_sst', 'es_ancora', 'sv_lines', 'sv_talbanken', 'tr_imst', 'uk_iu', 'hsb_ufal', 'ur_udtb', 'ug_udt', 'vi_vtb'];

// local DEV_LANGS = ['et_edt', 'ko_gsd', 'af_afribooms', 'hu_szeged', 'pl_sz', 'el_gdt', 'sk_snk', 'ja_gsd', 'sv_lines', 'grc_proiel', 'he_htb', 'fr_gsd', 'id_gsd', 'en_lines', 'no_bokmaal', 'la_ittb', 'fro_srcmf', 'cs_pdt', 'nl_alpino', 'fr_sequoia', 'fa_seraji', 'nl_lassysmall', 'fi_ftb', 'vi_vtb', 'ko_kaist', 'la_proiel', 'it_postwita', 'hr_set', 'cu_proiel', 'cs_fictree', 'sr_set', 'de_gsd', 'zh_gsd', 'da_ddt', 'fr_spoken', 'it_isdt', 'sv_talbanken', 'ro_rrt', 'es_ancora', 'eu_bdt', 'ar_padt', 'lv_lvtb', 'pl_lfg', 'sl_ssj', 'ug_udt', 'got_proiel', 'fi_tdt', 'ca_ancora', 'ru_syntagrus', 'pt_bosque', 'cs_cac', 'grc_perseus', 'bg_btb', 'uk_iu', 'ur_udtb', 'en_ewt', 'gl_ctg', 'no_nynorsk', 'hi_hdtb', 'en_gum', 'tr_imst'];
local DEV_LANGS = TRAIN_LANGS;

local TEST_LANGS = ['th_pud', 'la_perseus', 'gl_ctg', 'sv_lines', 'et_edt', 'cs_cac', 'pl_lfg', 'ru_taiga', 'lv_lvtb', 'ro_rrt', 'sl_sst', 'no_bokmaal', 'af_afribooms', 'fi_pud', 'el_gdt', 'ko_kaist', 'he_htb', 'pcm_nsc', 'sr_set', 'es_ancora', 'pl_sz', 'no_nynorsk', 'cs_pud', 'vi_vtb', 'nl_lassysmall', 'bg_btb', 'cu_proiel', 'en_lines', 'en_ewt', 'ca_ancora', 'fro_srcmf', 'la_proiel', 'sv_talbanken', 'nl_alpino', 'grc_perseus', 'it_isdt', 'ja_gsd', 'fo_oft', 'ur_udtb', 'da_ddt', 'gl_treegal', 'uk_iu', 'fi_tdt', 'ja_modern', 'sv_pud', 'ug_udt', 'kk_ktb', 'eu_bdt', 'br_keb', 'ga_idt', 'it_postwita', 'hr_set', 'ar_padt', 'fa_seraji', 'grc_proiel', 'sme_giella', 'hi_hdtb', 'ko_gsd', 'hsb_ufal', 'fr_gsd', 'pt_bosque', 'en_gum', 'kmr_mg', 'fr_spoken', 'tr_imst', 'sl_ssj', 'sk_snk', 'fr_sequoia', 'zh_gsd', 'hu_szeged', 'en_pud', 'ru_syntagrus', 'de_gsd', 'got_proiel', 'id_gsd', 'cs_fictree', 'la_ittb', 'fi_ftb', 'bxr_bdt', 'cs_pdt', 'no_nynorsklia', 'hy_armtdp'];

local READERS(xs, alternate=true) = {
    [x]: BASE_READER(x, alternate) for x in xs
};

local UD_ROOT = std.extVar("UD_ROOT");
local DATA_PATH(lang, split) = UD_ROOT + lang + "-ud-" + split + ".conllu";

{
    "dataset_readers": READERS(TRAIN_LANGS),
    // "validation_dataset_readers": READERS(DEV_LANGS, false),
    // "datasets_for_vocab_creation": ["train", "validation"],
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
    // "validation_iterator": {
    //     "type": "bucket",
    //     "sorting_keys": [["words", "roberta___mask"]],
    //     "batch_size": 8,
    //     "instances_per_epoch": 800,
    // },
    "model": {
        "type": "ud_biaffine_parser_multilang",
        "arc_representation_dim": 500,
        "dropout": 0.33,
        "word_dropout": 0.33,
        "input_dropout": 0.33,
        "encoder": {
            "type": "lstm",
            "bidirectional": true,
            "dropout": 0.33,
            "hidden_size": 200,
            "input_size": 818,
            "num_layers": 3
        },
        "langs_for_early_stop": TRAIN_LANGS,
        "pos_tag_embedding": {
            "embedding_dim": 50,
            "vocab_namespace": "pos"
        },
        "tag_representation_dim": 100,
        "model_name": MODEL_NAME,
        "text_field_embedder": {
            "token_embedders": {
                "roberta": {
                    "type": "transformer_pretrained_mismatched",
                    "model_name": "xlm-roberta-base",
                    "requires_grad": true,
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
    // "validation_data_paths": {
    //     [lang]: DATA_PATH(lang, "dev") for lang in DEV_LANGS
    // },
    "test_data_paths": {
        [lang]: DATA_PATH(lang, "test") for lang in TEST_LANGS
    },
    "trainer": {
        "type": "meta",
        "cuda_device": 0,
        "num_epochs": 40,
        "optimizer": {
          "type": "adam",
          "lr": 3e-5,
        },
        "patience": 10,
        "validation_metric": "+LAS_AVG",
        "num_gradient_accumulation_steps": 1,
        "tasks_per_step": 10,
        "wrapper": {
            "type": "fomaml",
            "optimizer_cls": "Adam",
            "optimizer_kwargs": {
                "lr": 3e-5
            }
        },
        "wandb": {
            "name": std.extVar("RUN_NAME"),
            "project": "allennlp-maml-parsing",
        },
    }
}
