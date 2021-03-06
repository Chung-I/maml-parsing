#!/usr/bin/env python3

# Compatible with Python 2.7 and 3.2+, can be used either as a module
# or a standalone executable.
#
# Copyright 2017, 2018 Institute of Formal and Applied Linguistics (UFAL),
# Faculty of Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Authors: Milan Straka, Martin Popel <surname@ufal.mff.cuni.cz>
#
# Changelog:
# - [12 Apr 2018] Version 0.9: Initial release.
# - [19 Apr 2018] Version 1.0: Fix bug in MLAS (duplicate entries in functional_children).
#                              Add --counts option.
# - [02 May 2018] Version 1.1: When removing spaces to match gold and system characters,
#                              consider all Unicode characters of category Zs instead of
#                              just ASCII space.
# - [25 Jun 2018] Version 1.2: Use python3 in the she-bang (instead of python).
#                              In Python2, make the whole computation use `unicode` strings.

# Command line usage
# ------------------
# conll18_ud_eval.py [-v] gold_conllu_file system_conllu_file
#
# - if no -v is given, only the official CoNLL18 UD Shared Task evaluation metrics
#   are printed
# - if -v is given, more metrics are printed (as precision, recall, F1 score,
#   and in case the metric is computed on aligned words also accuracy on these):
#   - Tokens: how well do the gold tokens match system tokens
#   - Sentences: how well do the gold sentences match system sentences
#   - Words: how well can the gold words be aligned to system words
#   - UPOS: using aligned words, how well does UPOS match
#   - XPOS: using aligned words, how well does XPOS match
#   - UFeats: using aligned words, how well does universal FEATS match
#   - AllTags: using aligned words, how well does UPOS+XPOS+FEATS match
#   - Lemmas: using aligned words, how well does LEMMA match
#   - UAS: using aligned words, how well does HEAD match
#   - LAS: using aligned words, how well does HEAD+DEPREL(ignoring subtypes) match
#   - CLAS: using aligned words with content DEPREL, how well does
#       HEAD+DEPREL(ignoring subtypes) match
#   - MLAS: using aligned words with content DEPREL, how well does
#       HEAD+DEPREL(ignoring subtypes)+UPOS+UFEATS+FunctionalChildren(DEPREL+UPOS+UFEATS) match
#   - BLEX: using aligned words with content DEPREL, how well does
#       HEAD+DEPREL(ignoring subtypes)+LEMMAS match
# - if -c is given, raw counts of correct/gold_total/system_total/aligned words are printed
#   instead of precision/recall/F1/AlignedAccuracy for all metrics.

# API usage
# ---------
# - load_conllu(file)
#   - loads CoNLL-U file from given file object to an internal representation
#   - the file object should return str in both Python 2 and Python 3
#   - raises UDError exception if the given file cannot be loaded
# - evaluate(gold_ud, system_ud)
#   - evaluate the given gold and system CoNLL-U files (loaded with load_conllu)
#   - raises UDError if the concatenated tokens of gold and system file do not match
#   - returns a dictionary with the metrics described above, each metric having
#     three fields: precision, recall and f1

# Description of token matching
# -----------------------------
# In order to match tokens of gold file and system file, we consider the text
# resulting from concatenation of gold tokens and text resulting from
# concatenation of system tokens. These texts should match -- if they do not,
# the evaluation fails.
#
# If the texts do match, every token is represented as a range in this original
# text, and tokens are equal only if their range is the same.

# Description of word matching
# ----------------------------
# When matching words of gold file and system file, we first match the tokens.
# The words which are also tokens are matched as tokens, but words in multi-word
# tokens have to be handled differently.
#
# To handle multi-word tokens, we start by finding "multi-word spans".
# Multi-word span is a span in the original text such that
# - it contains at least one multi-word token
# - all multi-word tokens in the span (considering both gold and system ones)
#   are completely inside the span (i.e., they do not "stick out")
# - the multi-word span is as small as possible
#
# For every multi-word span, we align the gold and system words completely
# inside this span using LCS on their FORMs. The words not intersecting
# (even partially) any multi-word span are then aligned as tokens.


from __future__ import division
from __future__ import print_function

import argparse
import io
import sys
import unicodedata
import unittest
from itertools import combinations
from copy import deepcopy
import numpy as np
import tqdm
import multiprocessing as mp
import psutil


def get_usable_cpu_count():
    p = psutil.Process()
    ret = 0
    with p.oneshot():
        ret = len(p.cpu_affinity())
    return ret

USABLE_CPU_COUNT = get_usable_cpu_count()

def get_bar(*args, **kwargs):
    return tqdm.tqdm(*args, ncols=80, **kwargs)

def mp_progress_map(func, arg_iter, num_workers=USABLE_CPU_COUNT, disable_tqdm=False):
    rs = []
    pool = mp.Pool(processes=num_workers)
    for args in arg_iter:
        rs.append(pool.apply_async(func, args))
    pool.close()

    rets = []
    _get_bar =  (lambda x: x) if disable_tqdm else get_bar
    for r in _get_bar(rs):
        rets.append(r.get())
    pool.join()
    return rets

# CoNLL-U column names
ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)

# Content and functional relations
CONTENT_DEPRELS = {
    "nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp", "obl", "vocative",
    "expl", "dislocated", "advcl", "advmod", "discourse", "nmod", "appos",
    "nummod", "acl", "amod", "conj", "fixed", "flat", "compound", "list",
    "parataxis", "orphan", "goeswith", "reparandum", "root", "dep"
}

FUNCTIONAL_DEPRELS = {
    "aux", "cop", "mark", "det", "clf", "case", "cc"
}

UNIVERSAL_FEATURES = {
    "PronType", "NumType", "Poss", "Reflex", "Foreign", "Abbr", "Gender",
    "Animacy", "Number", "Case", "Definite", "Degree", "VerbForm", "Mood",
    "Tense", "Aspect", "Voice", "Evident", "Polarity", "Person", "Polite"
}

METRICS = {"UAS", "LAS"}

# UD Error is used when raising exceptions in this module
class UDError(Exception):
    pass

# Conversion methods handling `str` <-> `unicode` conversions in Python2
def _decode(text):
    return text if sys.version_info[0] >= 3 or not isinstance(text, str) else text.decode("utf-8")

def _encode(text):
    return text if sys.version_info[0] >= 3 or not isinstance(text, unicode) else text.encode("utf-8")

# Load given CoNLL-U file into internal representation
class UDRepresentation:
    def __init__(self):
        # Characters of all the tokens in the whole file.
        # Whitespace between tokens is not included.
        self.characters = []
        # List of UDSpan instances with start&end indices into `characters`.
        self.tokens = []
        # List of UDWord instances.
        self.words = []
        # List of UDSpan instances with start&end indices into `characters`.
        self.sentences = []
class UDSpan:
    def __init__(self, start, end):
        self.start = start
        # Note that self.end marks the first position **after the end** of span,
        # so we can use characters[start:end] or range(start, end).
        self.end = end
class UDWord:
    def __init__(self, span, columns, is_multiword):
        # Span of this word (or MWT, see below) within ud_representation.characters.
        self.span = span
        # 10 columns of the CoNLL-U file: ID, FORM, LEMMA,...
        self.columns = columns
        # is_multiword==True means that this word is part of a multi-word token.
        # In that case, self.span marks the span of the whole multi-word token.
        self.is_multiword = is_multiword
        # Reference to the UDWord instance representing the HEAD (or None if root).
        self.parent = None
        # List of references to UDWord instances representing functional-deprel children.
        self.functional_children = []
        # Only consider universal FEATS.
        self.columns[FEATS] = "|".join(sorted(feat for feat in columns[FEATS].split("|")
                                              if feat.split("=", 1)[0] in UNIVERSAL_FEATURES))
        # Let's ignore language-specific deprel subtypes.
        self.columns[DEPREL] = columns[DEPREL].split(":")[0]
        # Precompute which deprels are CONTENT_DEPRELS and which FUNCTIONAL_DEPRELS
        self.is_content_deprel = self.columns[DEPREL] in CONTENT_DEPRELS
        self.is_functional_deprel = self.columns[DEPREL] in FUNCTIONAL_DEPRELS

def load_conllu(file):
    # Internal representation classes

    ud = UDRepresentation()

    # Load the CoNLL-U file
    index, sentence_start = 0, None
    while True:
        line = file.readline()
        if not line:
            break
        line = _decode(line.rstrip("\r\n"))

        # Handle sentence start boundaries
        if sentence_start is None:
            # Skip comments
            if line.startswith("#"):
                continue
            # Start a new sentence
            ud.sentences.append(UDSpan(index, 0))
            sentence_start = len(ud.words)
        if not line:
            # Add parent and children UDWord links and check there are no cycles
            def process_word(word):
                if word.parent == "remapping":
                    raise UDError("There is a cycle in a sentence")
                if word.parent is None:
                    head = int(word.columns[HEAD])
                    if head < 0 or head > len(ud.words) - sentence_start:
                        raise UDError("HEAD '{}' points outside of the sentence".format(_encode(word.columns[HEAD])))
                    if head:
                        parent = ud.words[sentence_start + head - 1]
                        word.parent = "remapping"
                        process_word(parent)
                        word.parent = parent

            for word in ud.words[sentence_start:]:
                process_word(word)
            # func_children cannot be assigned within process_word
            # because it is called recursively and may result in adding one child twice.
            for word in ud.words[sentence_start:]:
                if word.parent and word.is_functional_deprel:
                    word.parent.functional_children.append(word)

            # Check there is a single root node
            if len([word for word in ud.words[sentence_start:] if word.parent is None]) != 1:
                raise UDError("There are multiple roots in a sentence")

            # End the sentence
            ud.sentences[-1].end = index
            sentence_start = None
            continue

        # Read next token/word
        columns = line.split("\t")
        if len(columns) < 10:
            raise UDError("The CoNLL-U line does not contain 10 tab-separated columns: '{}'".format(_encode(line)))

        # Skip empty nodes
        if "." in columns[ID]:
            continue

        # Delete spaces from FORM, so gold.characters == system.characters
        # even if one of them tokenizes the space. Use any Unicode character
        # with category Zs.
        columns[FORM] = "".join(filter(lambda c: unicodedata.category(c) != "Zs", columns[FORM]))
        if not columns[FORM]:
            raise UDError("There is an empty FORM in the CoNLL-U file")

        # Save token
        ud.characters.extend(columns[FORM])
        ud.tokens.append(UDSpan(index, index + len(columns[FORM])))
        index += len(columns[FORM])

        # Handle multi-word tokens to save word(s)
        if "-" in columns[ID]:
            try:
                start, end = map(int, columns[ID].split("-"))
            except:
                raise UDError("Cannot parse multi-word token ID '{}'".format(_encode(columns[ID])))

            for _ in range(start, end + 1):
                word_line = _decode(file.readline().rstrip("\r\n"))
                word_columns = word_line.split("\t")
                if len(word_columns) != 10:
                    raise UDError("The CoNLL-U line does not contain 10 tab-separated columns: '{}'".format(_encode(word_line)))
                ud.words.append(UDWord(ud.tokens[-1], word_columns, is_multiword=True))
        # Basic tokens/words
        else:
            try:
                word_id = int(columns[ID])
            except:
                raise UDError("Cannot parse word ID '{}'".format(_encode(columns[ID])))
            if word_id != len(ud.words) - sentence_start + 1:
                raise UDError("Incorrect word ID '{}' for word '{}', expected '{}'".format(
                    _encode(columns[ID]), _encode(columns[FORM]), len(ud.words) - sentence_start + 1))

            try:
                head_id = int(columns[HEAD])
            except:
                raise UDError("Cannot parse HEAD '{}'".format(_encode(columns[HEAD])))
            if head_id < 0:
                raise UDError("HEAD cannot be negative")

            ud.words.append(UDWord(ud.tokens[-1], columns, is_multiword=False))

    if sentence_start is not None:
        raise UDError("The CoNLL-U file does not end with empty line")

    return ud

class Score:
    def __init__(self, gold_total, system_total, correct, aligned_total=None):
        self.correct = correct
        self.gold_total = gold_total
        self.system_total = system_total
        self.aligned_total = aligned_total
        self.precision = correct / system_total if system_total else 0.0
        self.recall = correct / gold_total if gold_total else 0.0
        self.f1 = 2 * correct / (system_total + gold_total) if system_total + gold_total else 0.0
        self.aligned_accuracy = correct / aligned_total if aligned_total else aligned_total
class AlignmentWord:
    def __init__(self, gold_word, system_word):
        self.gold_word = gold_word
        self.system_word = system_word
class Alignment:
    def __init__(self, gold_words, system_words):
        self.gold_words = gold_words
        self.system_words = system_words
        self.matched_words = []
        self.matched_words_map = {}
    def append_aligned_words(self, gold_word, system_word):
        self.matched_words.append(AlignmentWord(gold_word, system_word))
        self.matched_words_map[system_word] = gold_word

def spans_score(gold_spans, system_spans):
    correct, gi, si = 0, 0, 0
    while gi < len(gold_spans) and si < len(system_spans):
        if system_spans[si].start < gold_spans[gi].start:
            si += 1
        elif gold_spans[gi].start < system_spans[si].start:
            gi += 1
        else:
            correct += gold_spans[gi].end == system_spans[si].end
            si += 1
            gi += 1

    return Score(len(gold_spans), len(system_spans), correct)

def alignment_score(alignment, key_fn=None, filter_fn=None):
    if filter_fn is not None:
        gold = sum(1 for gold in alignment.gold_words if filter_fn(gold))
        system = sum(1 for system in alignment.system_words if filter_fn(system))
        aligned = sum(1 for word in alignment.matched_words if filter_fn(word.gold_word))
    else:
        gold = len(alignment.gold_words)
        system = len(alignment.system_words)
        aligned = len(alignment.matched_words)

    if key_fn is None:
        # Return score for whole aligned words
        return Score(gold, system, aligned)

    def gold_aligned_gold(word):
        return word
    def gold_aligned_system(word):
        return alignment.matched_words_map.get(word, "NotAligned") if word is not None else None
    correct = 0
    for words in alignment.matched_words:
        if filter_fn is None or filter_fn(words.gold_word):
            if key_fn(words.gold_word, gold_aligned_gold) == key_fn(words.system_word, gold_aligned_system):
                correct += 1

    return Score(gold, system, correct, aligned)

def beyond_end(words, i, multiword_span_end):
    if i >= len(words):
        return True
    if words[i].is_multiword:
        return words[i].span.start >= multiword_span_end
    return words[i].span.end > multiword_span_end

def extend_end(word, multiword_span_end):
    if word.is_multiword and word.span.end > multiword_span_end:
        return word.span.end
    return multiword_span_end

def find_multiword_span(gold_words, system_words, gi, si):
    # We know gold_words[gi].is_multiword or system_words[si].is_multiword.
    # Find the start of the multiword span (gs, ss), so the multiword span is minimal.
    # Initialize multiword_span_end characters index.
    if gold_words[gi].is_multiword:
        multiword_span_end = gold_words[gi].span.end
        if not system_words[si].is_multiword and system_words[si].span.start < gold_words[gi].span.start:
            si += 1
    else: # if system_words[si].is_multiword
        multiword_span_end = system_words[si].span.end
        if not gold_words[gi].is_multiword and gold_words[gi].span.start < system_words[si].span.start:
            gi += 1
    gs, ss = gi, si

    # Find the end of the multiword span
    # (so both gi and si are pointing to the word following the multiword span end).
    while not beyond_end(gold_words, gi, multiword_span_end) or \
          not beyond_end(system_words, si, multiword_span_end):
        if gi < len(gold_words) and (si >= len(system_words) or
                                     gold_words[gi].span.start <= system_words[si].span.start):
            multiword_span_end = extend_end(gold_words[gi], multiword_span_end)
            gi += 1
        else:
            multiword_span_end = extend_end(system_words[si], multiword_span_end)
            si += 1
    return gs, ss, gi, si

def compute_lcs(gold_words, system_words, gi, si, gs, ss):
    lcs = [[0] * (si - ss) for i in range(gi - gs)]
    for g in reversed(range(gi - gs)):
        for s in reversed(range(si - ss)):
            if gold_words[gs + g].columns[FORM].lower() == system_words[ss + s].columns[FORM].lower():
                lcs[g][s] = 1 + (lcs[g+1][s+1] if g+1 < gi-gs and s+1 < si-ss else 0)
            lcs[g][s] = max(lcs[g][s], lcs[g+1][s] if g+1 < gi-gs else 0)
            lcs[g][s] = max(lcs[g][s], lcs[g][s+1] if s+1 < si-ss else 0)
    return lcs

def align_words(gold_words, system_words):
    alignment = Alignment(gold_words, system_words)

    gi, si = 0, 0
    while gi < len(gold_words) and si < len(system_words):
        if gold_words[gi].is_multiword or system_words[si].is_multiword:
            # A: Multi-word tokens => align via LCS within the whole "multiword span".
            gs, ss, gi, si = find_multiword_span(gold_words, system_words, gi, si)

            if si > ss and gi > gs:
                lcs = compute_lcs(gold_words, system_words, gi, si, gs, ss)

                # Store aligned words
                s, g = 0, 0
                while g < gi - gs and s < si - ss:
                    if gold_words[gs + g].columns[FORM].lower() == system_words[ss + s].columns[FORM].lower():
                        alignment.append_aligned_words(gold_words[gs+g], system_words[ss+s])
                        g += 1
                        s += 1
                    elif lcs[g][s] == (lcs[g+1][s] if g+1 < gi-gs else 0):
                        g += 1
                    else:
                        s += 1
        else:
            # B: No multi-word token => align according to spans.
            if (gold_words[gi].span.start, gold_words[gi].span.end) == (system_words[si].span.start, system_words[si].span.end):
                alignment.append_aligned_words(gold_words[gi], system_words[si])
                gi += 1
                si += 1
            elif gold_words[gi].span.start <= system_words[si].span.start:
                gi += 1
            else:
                si += 1

    return alignment

METRIC_FUNCS = {
    "UAS": lambda w: w.columns[HEAD],
    "LAS": lambda w: (w.columns[HEAD], w.columns[DEPREL]),
}

def pairwise_score(gold_words, system_words, key_function):
    #assert gold_word.columns[FORM] == system_word.columns[FORM]
    #assert gold_word.span.start == gold_word.span.start
    #assert gold_word.span.end == gold_word.span.end
    correctness = [int(key_function(gold_word) == key_function(system_word))
                   for gold_word, system_word in zip(gold_words, system_words)]
    return correctness

def get_alignment(gold_ud, system_ud):
    # Check that the underlying character sequences do match.
    if gold_ud.characters != system_ud.characters:
        index = 0
        while index < len(gold_ud.characters) and index < len(system_ud.characters) and \
                gold_ud.characters[index] == system_ud.characters[index]:
            index += 1

        raise UDError(
            "The concatenation of tokens in gold file and in system file differ!\n" +
            "First 20 differing characters in gold file: '{}' and system file: '{}'".format(
                "".join(map(_encode, gold_ud.characters[index:index + 20])),
                "".join(map(_encode, system_ud.characters[index:index + 20]))
            )
        )

    # Align words
    alignment = align_words(gold_ud.words, system_ud.words)
    return alignment

def mean(l):
    return sum(l) / len(l)


def load_conllu_file(path):
    _file = open(path, mode="r", **({"encoding": "utf-8"} if sys.version_info >= (3, 0) else {}))
    return load_conllu(_file)

def paired_permutation_test(correctness_a, correctness_b):

    paired_correctness = {metric: list(zip(*[(a, b) if p > 0.5 else (b, a)
                                             for a, b, p in zip(correctness_a[metric], correctness_b[metric],
                                                                np.random.uniform(size=len(correctness_a[metric])))]))
                          for metric in METRICS}
    shuffled_diffs = {metric: mean(paired_correctness[metric][0]) - mean(paired_correctness[metric][1])
                      for metric in METRICS}
    return shuffled_diffs

def evaluate_wrapper(gold_file, system_file_a, system_file_b, n_trials, disable_tqdm,
                     skip_if_less):
    # Load CoNLL-U files
    gold_ud = load_conllu_file(gold_file)
    system_ud_a = load_conllu_file(system_file_a)
    system_ud_b = load_conllu_file(system_file_b)
    signif_cnts = {metric: [] for metric in METRICS}
    correctness_a = {metric: pairwise_score(gold_ud.words, system_ud_a.words, METRIC_FUNCS[metric])
                     for metric in METRICS}
    correctness_b = {metric: pairwise_score(gold_ud.words, system_ud_b.words, METRIC_FUNCS[metric])
                     for metric in METRICS}
    diff = {metric: mean(correctness_a[metric]) - mean(correctness_b[metric])
            for metric in METRICS}
    if skip_if_less and all([diff[metric] < 0 for metric in METRICS]):
        return {metric: 1.0 for metric in METRICS}

    #shuffled_diffs = []
    #for i in tqdm.tqdm(range(n_trials)):
    #    shuffled_diffs.append(paired_permutation_test(correctness_a, correctness_b))
    shuffled_diffs = mp_progress_map(paired_permutation_test, 
                                     [(correctness_a, correctness_b) for i in range(n_trials)],
                                     disable_tqdm=disable_tqdm)

    significances = {metric: (sum([shuffled_diff[metric] > diff[metric] for shuffled_diff in shuffled_diffs]) + 1)/(n_trials+1)
                     for metric in METRICS}
    return significances

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_file", type=str,
                        help="Name of the CoNLL-U file with the gold data.")
    parser.add_argument("system_files", type=str, nargs='+',
                        help="Name of the CoNLL-U file with the predicted data.")
    parser.add_argument("--n-trials", type=int, default=10,
                        help="number of samples of permutations")
    parser.add_argument("--verbose", "-v", default=False, action="store_true",
                        help="Print all metrics.")
    parser.add_argument("--skip-if-less", action='store_true',
                        help='skip if former system file performs worse than latter')
    parser.add_argument("--counts", "-c", default=False, action="store_true",
                        help="Print raw counts of correct/gold/system/aligned words instead of prec/rec/F1 for all metrics.")
    parser.add_argument("--disable-tqdm", action='store_true', help='disable tqdm if specified.')
    args = parser.parse_args()

    # Evaluate
    for system_file_a, system_file_b in combinations(args.system_files, 2):
        significances = evaluate_wrapper(args.gold_file, system_file_a, system_file_b,
                                         args.n_trials, args.disable_tqdm, args.skip_if_less)
        print(significances)

if __name__ == "__main__":
    main()

# Tests, which can be executed with `python -m unittest conll18_ud_eval`.
class TestAlignment(unittest.TestCase):
    @staticmethod
    def _load_words(words):
        """Prepare fake CoNLL-U files with fake HEAD to prevent multiple roots errors."""
        lines, num_words = [], 0
        for w in words:
            parts = w.split(" ")
            if len(parts) == 1:
                num_words += 1
                lines.append("{}\t{}\t_\t_\t_\t_\t{}\t_\t_\t_".format(num_words, parts[0], int(num_words>1)))
            else:
                lines.append("{}-{}\t{}\t_\t_\t_\t_\t_\t_\t_\t_".format(num_words + 1, num_words + len(parts) - 1, parts[0]))
                for part in parts[1:]:
                    num_words += 1
                    lines.append("{}\t{}\t_\t_\t_\t_\t{}\t_\t_\t_".format(num_words, part, int(num_words>1)))
        return load_conllu((io.StringIO if sys.version_info >= (3, 0) else io.BytesIO)("\n".join(lines+["\n"])))

    def _test_exception(self, gold, system):
        self.assertRaises(UDError, evaluate, self._load_words(gold), self._load_words(system))

    def _test_ok(self, gold, system, correct):
        metrics = evaluate(self._load_words(gold), self._load_words(system))
        gold_words = sum((max(1, len(word.split(" ")) - 1) for word in gold))
        system_words = sum((max(1, len(word.split(" ")) - 1) for word in system))
        self.assertEqual((metrics["Words"].precision, metrics["Words"].recall, metrics["Words"].f1),
                         (correct / system_words, correct / gold_words, 2 * correct / (gold_words + system_words)))

    def test_exception(self):
        self._test_exception(["a"], ["b"])

    def test_equal(self):
        self._test_ok(["a"], ["a"], 1)
        self._test_ok(["a", "b", "c"], ["a", "b", "c"], 3)

    def test_equal_with_multiword(self):
        self._test_ok(["abc a b c"], ["a", "b", "c"], 3)
        self._test_ok(["a", "bc b c", "d"], ["a", "b", "c", "d"], 4)
        self._test_ok(["abcd a b c d"], ["ab a b", "cd c d"], 4)
        self._test_ok(["abc a b c", "de d e"], ["a", "bcd b c d", "e"], 5)

    def test_alignment(self):
        self._test_ok(["abcd"], ["a", "b", "c", "d"], 0)
        self._test_ok(["abc", "d"], ["a", "b", "c", "d"], 1)
        self._test_ok(["a", "bc", "d"], ["a", "b", "c", "d"], 2)
        self._test_ok(["a", "bc b c", "d"], ["a", "b", "cd"], 2)
        self._test_ok(["abc a BX c", "def d EX f"], ["ab a b", "cd c d", "ef e f"], 4)
        self._test_ok(["ab a b", "cd bc d"], ["a", "bc", "d"], 2)
        self._test_ok(["a", "bc b c", "d"], ["ab AX BX", "cd CX a"], 1)
