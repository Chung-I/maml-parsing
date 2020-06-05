from typing import Dict, Tuple, List, Iterator, Any, Callable
from collections import OrderedDict
import logging
import itertools
import glob
import os
import numpy as np
import re

from overrides import overrides
from conllu import parse_incr
from src.data.dataset_readers.parser import parse_line, DEFAULT_FIELDS
from src.data.dataset_readers.util import generate_stack_inputs, ud_v1_to_v2_conversion

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.fields import IndexField, ListField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

BLANK_FILES = ['ja_bccwj-ud-train',
               'ar_nyuad-ud-train',
               'en_esl-ud-train',
               'fr_ftb-ud-train',
               'qhe_heincs-ud-train']

logger = logging.getLogger(__name__)


def lazy_parse(text: str, fields: Tuple[str, ...]=DEFAULT_FIELDS):
    for sentence in text.split("\n\n"):
        if sentence:
            # TODO: upgrade conllu library
            yield [parse_line(line, fields)
                   for line in sentence.split("\n")
                   if line and not line.strip().startswith("#")]


def get_file_paths(pathname: str, languages: List[str]):
    """
    Gets a list of all files by the pathname with the given language ids.
    Filenames are assumed to have the language identifier followed by a dash
    as a prefix (e.g. en-universal.conll).

    # Parameters

    pathname :  `str`, required.
        An absolute or relative pathname (can contain shell-style wildcards)
    languages : `List[str]`, required
        The language identifiers to use.

    # Returns

    A list of tuples (language id, file path).
    """
    paths = []
    languages = [re.sub("[0-9]", "", l) for l in languages]
    lang = languages[0]
    delimiter = "-" if lang.split("_")[0] != lang else "_"
    for file_path in glob.glob(pathname):
        base = os.path.splitext(os.path.basename(file_path))[0]
        if base in BLANK_FILES:
            logger.info(
                "Skipping %s language at %s since it has no text", lang, file_path
            )
            continue
        lang_id = re.sub("[0-9]", "", base.split(delimiter)[0])
        if lang_id == base:
            lang_id = base.split("-")[0]
        if lang_id in languages:
            paths.append((lang_id, file_path))

    if not paths:
        raise ConfigurationError("No dataset files to read")

    return paths



@DatasetReader.register("ud_multilang")
class UniversalDependenciesMultiLangDatasetReader(DatasetReader):
    """
    Reads multiple files in the conllu Universal Dependencies format.
    All files should be in the same directory and the filenames should have
    the language identifier followed by a dash as a prefix (e.g. en-universal.conll)
    When using the alternate option, the reader alternates randomly between
    the files every instances_per_file. The is_first_pass_for_vocab disables
    this behaviour for the first pass (could be useful for a single full path
    over the dataset in order to generate a vocabulary).

    Notice: when using the alternate option, one should also use the `instances_per_epoch`
    option for the iterator. Otherwise, each epoch will loop infinitely.

    # Parameters

    languages : `List[str]`, required
        The language identifiers to use.
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        The token indexers to be applied to the words TextField.
    use_language_specific_pos : `bool`, optional (default = False)
        Whether to use UD POS tags, or to use the language specific POS tags
        provided in the conllu format.
    alternate : `bool`, optional (default = True)
        Whether to alternate between input files.
    is_first_pass_for_vocab : `bool`, optional (default = True)
        Whether the first pass will be for generating the vocab. If true,
        the first pass will run over the entire dataset of each file (even if alternate is on).
    instances_per_file : `int`, optional (default = 32)
        The amount of consecutive cases to sample from each input file when alternating.
    use_language_specific_deprel: `bool`, optional (default = False)
        Whether to use language-specific relations or not. If true,
        language-specific parts of the relation, e.g. "obl:appl", will be truncated
        to "obl".
    """

    def __init__(
        self,
        languages: List[str],
        token_indexers: Dict[str, TokenIndexer] = None,
        use_language_specific_pos: bool = False,
        use_lemma: bool = False,
        use_ufeat: bool = False,
        alternate: bool = True,
        is_first_pass_for_vocab: bool = True,
        instances_per_file: int = 32,
        read_dependencies: bool = True,
        read_language: bool = True,
        view: str = 'graph',
        use_language_specific_deprel: bool = True,
        deprel_file: str = None,
        in_memory: bool = False,
        max_len: int = 0, 
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if view not in ["graph", "transition"]:
            raise NotImplementedError

        self._languages = languages
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._use_language_specific_pos = use_language_specific_pos
        self._use_lemma = use_lemma
        self._use_ufeat = use_ufeat
        self._view = view

        self._is_first_pass_for_vocab = is_first_pass_for_vocab
        self._alternate = alternate
        self._instances_per_file = instances_per_file

        self._is_first_pass = True
        self._iterators: List[Tuple[str, Iterator[Any]]] = None
        self._read_dependencies = read_dependencies
        self._read_language = read_language
        self._use_language_specific_deprel = use_language_specific_deprel
        self._deprels = None
        self._max_len = max_len 
        self._in_memory = in_memory
        self._file_cache = None
        if self._in_memory:
            self._file_cache = {}
        if deprel_file is not None:
            with open(deprel_file) as fp:
                self._deprels = fp.read().splitlines()

    def map_deprel(self, rel):
        rel = ud_v1_to_v2_conversion(rel)
        urel = rel.split(":")[0]
        xrel = rel

        if self._deprels is None:
            if self._use_language_specific_deprel:
                return xrel
            else:
                return urel
        else:
            if self._use_language_specific_deprel and rel in self._deprels:
                return xrel
            else:
                return urel
            return rel

    def _read_one_file(self, lang: str, file_path: str):
        try:
            conllu_text = self._file_cache[file_path]
            logger.info(
                "Reading UD instances for %s language from cached conllu dataset at: %s", lang, file_path
            )
        except (TypeError, KeyError) as error:
            with open(file_path, "r") as conllu_file:
                conllu_text = conllu_file.read()
            logger.info(
                "Reading UD instances for %s language from conllu dataset at: %s", lang, file_path
            )
            if isinstance(error, KeyError):
                self._file_cache[file_path] = conllu_text

        for annotation in lazy_parse(conllu_text):
            # CoNLLU annotations sometimes add back in words that have been elided
            # in the original sentence; we remove these, as we're just predicting
            # dependencies for the original sentence.
            # We filter by None here as elided words have a non-integer word id,
            # and are replaced with None by the conllu python library.
            multiword_tokens = [x for x in annotation if x["multi_id"] is not None]
            annotation = [x for x in annotation if x["id"] is not None]
            if self._max_len and len(annotation) > self._max_len:
                logger.info(
                    f"sentence length {len(annotation)} longer than {self._max_len}; skipping"
                )
                continue 

            if len(annotation) == 0:
                continue
    
            def get_field(tag: str, map_fn: Callable[[Any], Any] = None) -> List[Any]:
                map_fn = map_fn if map_fn is not None else lambda x: x
                return [map_fn(x[tag]) if x[tag] is not None else "_" for x in annotation if tag in x]
    
            # Extract multiword token rows (not used for prediction, purely for evaluation)
            ids = [x["id"] for x in annotation]
            multiword_ids = [x["multi_id"] for x in multiword_tokens]
            multiword_forms = [x["form"] for x in multiword_tokens]
    
            words = get_field("form")
            lemmas = get_field("lemma")
            upos_tags = get_field("upostag")
            xpos_tags = get_field("xpostag")
            feats = get_field("feats", lambda x: "|".join(k + "=" + v for k, v in x.items())
                                                 if hasattr(x, "items") else "_")
            heads = get_field("head")
            dep_rels = get_field("deprel")
            dep_rels = list(map(self.map_deprel, dep_rels))
            dependencies = list(zip(dep_rels, heads)) if self._read_dependencies else None
            yield self.text_to_instance(lang, words, lemmas, upos_tags, xpos_tags,
                                        feats, dependencies, ids, multiword_ids, multiword_forms)

    @overrides
    def _read(self, file_path: str):
        file_paths = get_file_paths(file_path, self._languages)
        if (self._is_first_pass and self._is_first_pass_for_vocab) or (not self._alternate):
            iterators = [
                iter(self._read_one_file(lang, file_path)) for (lang, file_path) in file_paths
            ]
            self._is_first_pass = False
            for inst in itertools.chain(*iterators):
                yield inst

        else:
            if self._iterators is None:
                self._iterators = [
                    (lang, iter(self._read_one_file(lang, file_path)))
                    for (lang, file_path) in file_paths
                ]
            num_files = len(file_paths)
            while True:
                ind = np.random.randint(num_files)
                lang, lang_iter = self._iterators[ind]
                for _ in range(self._instances_per_file):
                    try:
                        yield lang_iter.__next__()
                    except StopIteration:
                        lang, file_path = file_paths[ind]
                        lang_iter = iter(self._read_one_file(lang, file_path))
                        self._iterators[ind] = (lang, lang_iter)
                        yield lang_iter.__next__()

    def _text_to_graph_instance(self,
                                tokens: TextField,
                                dependencies: List[Tuple[str, int]] = None):
        fields: Dict[str, Field] = {}

        if dependencies is not None:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            fields["head_tags"] = SequenceLabelField([x[0] for x in dependencies],
                                                     tokens,
                                                     label_namespace="head_tags")
            fields["head_indices"] = SequenceLabelField([int(x[1]) for x in dependencies],
                                                        tokens,
                                                        label_namespace="head_index_tags")
        return fields

    def _text_to_transition_instance(self,
                                     tokens,
                                     dependencies):

        fields: Dict[str, Field] = {}
        head_ids = [int(x[1]) for x in dependencies]
        head_tags = [x[0] for x in dependencies]
        stacked_head_ids, children, siblings, stacked_head_tags, _ = \
            generate_stack_inputs([0] + head_ids, ['root'] + head_tags, 'inside_out')

        fields["head_indices"] = SequenceLabelField(head_ids,
                                                    tokens,
                                                    label_namespace="head_index_tags")
        fields["head_tags"] = SequenceLabelField(head_tags,
                                                 tokens,
                                                 label_namespace="head_tags")
        stacked_head_indices = \
            ListField([IndexField(idx, tokens) for idx in stacked_head_ids])
        fields["stacked_head_indices"] = stacked_head_indices
        fields["stacked_head_tags"] = SequenceLabelField(stacked_head_tags,
                                                         stacked_head_indices,
                                                         label_namespace="head_tags")
        fields["children"] = \
            ListField([IndexField(idx, tokens) for idx in children])
        fields["siblings"] = \
            ListField([IndexField(idx, tokens) for idx in siblings])

        return fields

    @overrides
    def text_to_instance(self,  # type: ignore
                         lang: str,
                         words: List[str],
                         lemmas: List[str] = None,
                         upos_tags: List[str] = None,
                         xpos_tags: List[str] = None,
                         feats: List[str] = None,
                         dependencies: List[Tuple[str, int]] = None,
                         ids: List[str] = None,
                         multiword_ids: List[str] = None,
                         multiword_forms: List[str] = None) -> Instance:

        """
        # Parameters

        lang : `str`, required.
            The language identifier.
        words : `List[str]`, required.
            The words in the sentence to be encoded.
        upos_tags : `List[str]`, required.
            The universal dependencies POS tags for each word.
        dependencies `List[Tuple[str, int]]`, optional (default = None)
            A list of  (head tag, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that word being the root of
            the dependency tree.

        # Returns

        An instance containing words, upos tags, dependency head tags and head
        indices as fields. The language identifier is stored in the metadata.
        """
        tokens = TextField([Token(w) for w in words], self._token_indexers)
        if self._view == "graph":
            fields = self._text_to_graph_instance(tokens, dependencies)
        elif self._view == "transition":
            fields = self._text_to_transition_instance(tokens, dependencies)

        fields["words"] = tokens
        fields["pos_tags"] = SequenceLabelField(upos_tags, tokens, label_namespace="pos")
        if self._read_language:
            fields["langs"] = LabelField(lang, label_namespace="lang_labels")

        fields["metadata"] = MetadataField({
            "words": words,
            "upos": upos_tags,
            "xpos": xpos_tags,
            "feats": feats,
            "lemmas": lemmas,
            "ids": ids,
            "multiword_ids": multiword_ids,
            "multiword_forms": multiword_forms,
            "lang": lang,
            "gold_tags": [x[0] for x in dependencies] if dependencies else None,
            "gold_heads": [x[1] for x in dependencies] if dependencies else None,
        })

        return Instance(fields)
