from typing import Dict, Optional, Any, List, Tuple
import logging
import copy

from collections import defaultdict
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules import Dropout
import numpy

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding, InputVariationalDropout
from allennlp.modules import FeedForward, TimeDistributed, Seq2VecEncoder
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, Activation
from allennlp.nn.util import get_text_field_mask, get_range_vector
from allennlp.nn.util import (
    get_device_of,
    masked_log_softmax,
    masked_mean,
    get_lengths_from_binary_sequence_mask,
    batched_span_select,
    sequence_cross_entropy_with_logits,
)
from allennlp.nn.chu_liu_edmonds import decode_mst
from allennlp.training.metrics import AttachmentScores, Average, CategoricalAccuracy

from src.models.biaffine_dependency_parser import BiaffineDependencyParser
from src.modules.vib import ContinuousVIB
from src.modules.crf import log_partition
from src.training.util import get_lang_means, get_lang_mean

logger = logging.getLogger(__name__)

POS_TO_IGNORE = {"`", "''", ":", ",", ".", "PU", "PUNCT", "SYM"}


@Model.register("ud_biaffine_parser_multilang_vib")
class BiaffineDependencyParserMultiLangVIB(Model):
    """
    This dependency parser implements the multi-lingual extension
    of the Dozat and Manning (2016) model as described in
    [Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot
    Dependency Parsing (Schuster et al., 2019)] (https://www.aclweb.org/anthology/papers/N/N19/N19-1162).
    Also, please refer to the [alignment computation code]
    (https://github.com/TalSchuster/CrossLingualELMo).

    All parameters are shared across all languages except for
    the text_field_embedder. For aligned ELMo embeddings, use the
    elmo_token_embedder_multilang with the pre-computed alignments
    to the mutual embedding space.
    Also, the universal_dependencies_multilang dataset reader
    supports loading of multiple sources and storing the language
    identifier in the metadata.


    # Parameters

    vocab : `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : `TextFieldEmbedder`, required
        Used to embed the `tokens` `TextField` we get as input to the model.
    encoder : `Seq2SeqEncoder`
        The encoder (with its own internal stacking) that we will use to generate representations
        of tokens.
    tag_representation_dim : `int`, required.
        The dimension of the MLPs used for dependency tag prediction.
    arc_representation_dim : `int`, required.
        The dimension of the MLPs used for head arc prediction.
    tag_feedforward : `FeedForward`, optional, (default = None).
        The feedforward network used to produce tag representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    arc_feedforward : `FeedForward`, optional, (default = None).
        The feedforward network used to produce arc representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    pos_tag_embedding : `Embedding`, optional.
        Used to embed the `pos_tags` `SequenceLabelField` we get as input to the model.
    use_mst_decoding_for_validation : `bool`, optional (default = True).
        Whether to use Edmond's algorithm to find the optimal minimum spanning tree during validation.
        If false, decoding is greedy.
    langs_for_early_stop : `List[str]`, optional, (default = [])
        Which languages to include in the averaged metrics
        (that could be used for early stopping).
    dropout : `float`, optional, (default = 0.0)
        The variational dropout applied to the output of the encoder and MLP layers.
    input_dropout : `float`, optional, (default = 0.0)
        The dropout applied to the embedded text input.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    regularizer : `RegularizerApplicator`, optional (default=`None`)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        tag_representation_dim: int,
        arc_representation_dim: int,
        vib: Params = None,
        model_name: str = None,
        tag_feedforward: FeedForward = None,
        arc_feedforward: FeedForward = None,
        pos_tag_embedding: Embedding = None,
        use_mst_decoding_for_validation: bool = True,
        langs_for_early_stop: List[str] = None,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        word_dropout: float = 0.0,
        lexical_dropout: float = 0.0,
        dropout_location: str = 'input',
        pos_dropout: float = 0.0,
        max_sent_len: int = 512,
        tag_dim: int = 128,
        per_lang_vib: bool = True,
        adv_layer: str = 'encoder',
        inspect_layer: str = 'encoder',
        lang_mean_regex: str = None,
        ft_lang_mean_dir: str = None,
        zs_lang_mean_dir: Optional[List[str]] = None,
        typo_encoder: Seq2VecEncoder = None,
        typo_feedforward: FeedForward = None,
        predict_pos: bool = False,
        token_embedder_key: str = None,
        use_crf: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder

        if model_name:
            from src.data.token_indexers import PretrainedAutoTokenizer
            self._tokenizer = PretrainedAutoTokenizer.load(model_name)

        encoder_dim = tag_dim

        self.head_arc_feedforward = arc_feedforward or FeedForward(
            encoder_dim, 1, arc_representation_dim, Activation.by_name("elu")()
        )
        self.child_arc_feedforward = copy.deepcopy(self.head_arc_feedforward)

        self.arc_attention = BilinearMatrixAttention(
            arc_representation_dim, arc_representation_dim, use_input_biases=True
        )

        num_labels = self.vocab.get_vocab_size("head_tags")

        self.head_tag_feedforward = tag_feedforward or FeedForward(
            encoder_dim, 1, tag_representation_dim, Activation.by_name("elu")()
        )
        self.child_tag_feedforward = copy.deepcopy(self.head_tag_feedforward)

        self.tag_bilinear = torch.nn.modules.Bilinear(
            tag_representation_dim, tag_representation_dim, num_labels
        )

        self._pos_tag_embedding = pos_tag_embedding or None
        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)
        self._word_dropout = word_dropout
        self._lexical_dropout = lexical_dropout if lexical_dropout > 0 else word_dropout
        self._pos_dropout = pos_dropout if pos_dropout > 0 else word_dropout
        assert dropout_location in ['input', 'lm']
        self._dropout_location = dropout_location

        self._head_sentinel = torch.nn.Parameter(torch.randn([1, 1, tag_dim]))

        representation_dim = text_field_embedder.get_output_dim()
        if pos_tag_embedding is not None:
            representation_dim += pos_tag_embedding.get_output_dim()

        # check_dimensions_match(
        #     representation_dim,
        #     encoder.get_input_dim(),
        #     "text field embedding dim",
        #     "encoder input dim",
        # )

        check_dimensions_match(
            tag_representation_dim,
            self.head_tag_feedforward.get_output_dim(),
            "tag representation dim",
            "tag feedforward output dim",
        )
        check_dimensions_match(
            arc_representation_dim,
            self.head_arc_feedforward.get_output_dim(),
            "arc representation dim",
            "arc feedforward output dim",
        )

        self.use_mst_decoding_for_validation = use_mst_decoding_for_validation

        tags = self.vocab.get_token_to_index_vocabulary("pos")
        punctuation_tag_indices = {
            tag: index for tag, index in tags.items() if tag in POS_TO_IGNORE
        }
        self._pos_to_ignore = set(punctuation_tag_indices.values())
        logger.info(
            f"Found POS tags corresponding to the following punctuation : {punctuation_tag_indices}. "
            "Ignoring words with these POS tags for evaluation."
        )

        self._lang_kl_divs: Dict[str, Average] = defaultdict(Average)
        self._lang_attachment_scores: Dict[str, AttachmentScores] = defaultdict(AttachmentScores)

        self._langs_for_early_stop = langs_for_early_stop or []

        num_langs = self.vocab.get_vocab_size(namespace="lang_labels")

        self._per_lang_vib = per_lang_vib
        if self._per_lang_vib:
            self.r_mean = torch.nn.Parameter(torch.randn(num_langs, max_sent_len, tag_dim))
            self.r_std = torch.nn.Parameter(torch.randn(num_langs, max_sent_len, tag_dim))
        else:
            self.r_mean = torch.nn.Parameter(torch.randn(max_sent_len, tag_dim))
            self.r_std = torch.nn.Parameter(torch.randn(max_sent_len, tag_dim))

        self._lang_means = None
        if lang_mean_regex is not None:
            lang_means = get_lang_means(lang_mean_regex, self.vocab)
            self._lang_means = torch.nn.Parameter(lang_means, requires_grad=False)

        self._ft_lang_mean = None
        if ft_lang_mean_dir is not None:
            ft_lang_mean = get_lang_mean(ft_lang_mean_dir)
            self._ft_lang_mean = torch.nn.Parameter(ft_lang_mean, requires_grad=False)
            
        self._zs_lang_mean = None
        if zs_lang_mean_dir is not None:
            anchor_lang_mean = get_lang_mean(zs_lang_mean_dir[0])
            zs_lang_mean = get_lang_mean(zs_lang_mean_dir[1])
            self._zs_lang_mean = zs_lang_mean - anchor_lang_mean# - zs_lang_mean

        assert adv_layer in ['vib', 'encoder']
        self._adv_layer = adv_layer

        assert inspect_layer in ['embedding', 'vib', 'encoder', 'projection']
        self._inspect_layer = inspect_layer

        self.typo_encoder = typo_encoder
        self.typo_feedforward = typo_feedforward

        self._predict_pos = predict_pos
        self._num_pos_tags = self.vocab.get_vocab_size("pos")
        self.tag_projection_layer = None
        self._token_embedder_key = token_embedder_key
        if predict_pos:
            self.tag_projection_layer = TimeDistributed(
                torch.nn.Linear(tag_dim, self._num_pos_tags)
            )
            self._pos_metrics = {
                "pos_accuracy": CategoricalAccuracy(),
            }

        self.VIB = None
        if vib is not None:
            self.VIB = ContinuousVIB.from_params(
                Params(vib),
                embedding_dim=representation_dim,
            )

        self.use_crf = use_crf
        
        initializer(self)

    def _embed(
        self,
        words: TextFieldTensors,
        pos_tags: torch.LongTensor,
        mask: torch.Tensor,
        metadata: List[Dict[str, Any]],
        lemmas: TextFieldTensors = None,
        feats: TextFieldTensors = None,
        langs: torch.LongTensor = None,
        batch_lang: str = None,
    ) -> torch.Tensor:

        if self._dropout_location == 'input':
            words[self._token_embedder_key]["token_ids"] = self._apply_token_dropout(
                words[self._token_embedder_key]["token_ids"],
                words[self._token_embedder_key]["mask"],
                self._lexical_dropout,
                words[self._token_embedder_key]["offsets"],
                form='subword',
            )

        embedded_text_input = self.text_field_embedder(words, lang=batch_lang)
        if self._lang_means is not None or self._ft_lang_mean is not None or self._zs_lang_mean is not None:
            batch_size, seq_len, _ = embedded_text_input.size()
            lang_mean = self._zs_lang_mean if self._zs_lang_mean is not None else self._ft_lang_mean
            if langs is None:
                means = lang_mean.view(1, 1, -1).repeat(batch_size, seq_len, 1)
            else:
                expanded_langs = langs.unsqueeze(-1).repeat(1, seq_len)
                means = self._lang_means[expanded_langs] 
            embedded_text_input = embedded_text_input - means.to(embedded_text_input.device)
        if self._dropout_location == 'lm':
            embedded_text_input = self._apply_token_dropout(embedded_text_input,
                                                            mask,
                                                            self._lexical_dropout,
                                                            form='tensor')
        if pos_tags is not None and self._pos_tag_embedding is not None:
            pos_tags = self._apply_token_dropout(pos_tags,
                                                 mask,
                                                 self._pos_dropout,
                                                 form='word')
            embedded_pos_tags = self._pos_tag_embedding(pos_tags)
            embedded_text_input = torch.cat([embedded_text_input, embedded_pos_tags], -1)
        elif self._pos_tag_embedding is not None:
            raise ConfigurationError("Model uses a POS embedding, but no POS tags were passed.")

        embedded_text_input = self._input_dropout(embedded_text_input)

        return embedded_text_input

    def _bottleneck(
        self,
        embedded_text_input: torch.Tensor,
        pos_tags: torch.LongTensor,
        mask: torch.Tensor,
        head_tags: torch.LongTensor = None,
        head_indices: torch.LongTensor = None,
        langs: torch.LongTensor = None,
        variational: bool = False,
    ):
        sample_method = "iid" if variational else "argmax"
        sample_size = None if variational else 1

        if self._per_lang_vib:
            r_mean = self.r_mean[langs]
            r_std = self.r_std[langs]
        else:
            bsz = embedded_text_input.size(0)
            r_mean = self.r_mean.unsqueeze(0).repeat(bsz, 1, 1)
            r_std = self.r_std.unsqueeze(0).repeat(bsz, 1, 1) 

        embedded_text_input, head_indices, head_tags, pos_tags, mask, kl_loss, kl_div, kl_div2 = self.VIB(
            head_indices, head_tags, pos_tags, mask, r_mean, r_std, sample_size=sample_size,
            sample_method=sample_method, type_embeds=embedded_text_input,
        )
        return embedded_text_input, head_indices, head_tags, pos_tags, mask, \
            kl_loss, kl_div, kl_div2

    def _gen_typo_feats(self,
                        embedded_text_input: torch.Tensor,
                        mask: torch.LongTensor) -> torch.Tensor:
        typo_feats = self.typo_encoder(embedded_text_input, mask)
        typo_feats = self.typo_feedforward(typo_feats)
        batch_size, seq_len = mask.size()
        pooled_feats = torch.mean(typo_feats, dim=0)
        expanded_feats = pooled_feats.view(1, 1, -1).expand(batch_size, seq_len, -1)

        return expanded_feats

    def _project(
        self,
        encoded_text: torch.Tensor,
        mask: torch.LongTensor,
        head_tags: torch.LongTensor = None,
        head_indices: torch.LongTensor = None,
        return_arc_representation: bool = False,
    ):
        batch_size, _, encoding_dim = encoded_text.size()

        head_sentinel = self._head_sentinel.expand(batch_size, 1, encoding_dim)
        # Concatenate the head sentinel onto the sentence representation.
        encoded_text = torch.cat([head_sentinel, encoded_text], 1)
        mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)
        if head_indices is not None:
            head_indices = torch.cat([head_indices.new_zeros(batch_size, 1), head_indices], 1)
        if head_tags is not None:
            head_tags = torch.cat([head_tags.new_zeros(batch_size, 1), head_tags], 1)
        float_mask = mask.float()
        encoded_text = self._dropout(encoded_text)

        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc_representation = self._dropout(self.head_arc_feedforward(encoded_text))
        child_arc_representation = self._dropout(self.child_arc_feedforward(encoded_text))

        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag_representation = self._dropout(self.head_tag_feedforward(encoded_text))
        child_tag_representation = self._dropout(self.child_tag_feedforward(encoded_text))
        # shape (batch_size, sequence_length, sequence_length)
        attended_arcs = self.arc_attention(head_arc_representation, child_arc_representation)

        minus_inf = -1e8
        minus_mask = (1 - float_mask) * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)
        seq_len = mask.size(1)
        root_mask = torch.cat([mask.new_ones(batch_size, 1),
                               mask.new_zeros(batch_size, seq_len - 1)], dim=1)
        attended_arcs.masked_fill_(root_mask.bool().unsqueeze(2), minus_inf)

        if return_arc_representation:
            return head_arc_representation, child_arc_representation, \
                head_tag_representation, child_tag_representation, \
                attended_arcs, mask, head_tags, head_indices
        return head_tag_representation, child_tag_representation, attended_arcs, \
            mask, head_tags, head_indices

    def _attend_and_normalize(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        attended_arcs: torch.Tensor,
        mask: torch.Tensor,
    ):
        batch_size, sequence_length, tag_representation_dim = head_tag_representation.size()

        lengths = mask.data.sum(dim=1).long().cpu().numpy()

        expanded_shape = [batch_size, sequence_length, sequence_length, tag_representation_dim]
        head_tag_representation = head_tag_representation.unsqueeze(2)
        head_tag_representation = head_tag_representation.expand(*expanded_shape).contiguous()
        child_tag_representation = child_tag_representation.unsqueeze(1)
        child_tag_representation = child_tag_representation.expand(*expanded_shape).contiguous()
        # Shape (batch_size, sequence_length, sequence_length, num_head_tags)
        pairwise_head_logits = self.tag_bilinear(head_tag_representation, child_tag_representation)

        # Note that this log_softmax is over the tag dimension, and we don't consider pairs
        # of tags which are invalid (e.g are a pair which includes a padded element) anyway below.
        # Shape (batch, num_labels,sequence_length, sequence_length)
        normalized_pairwise_head_logits = F.log_softmax(pairwise_head_logits, dim=3).permute(
            0, 3, 1, 2
        )

        # Mask padded tokens, because we only want to consider actual words as heads.
        minus_inf = -1e8
        minus_mask = (1 - mask.float()) * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        # Shape (batch_size, sequence_length, sequence_length)
        if not self.use_crf:
            normalized_arc_logits = F.log_softmax(attended_arcs, dim=2).transpose(1, 2)
        else:
            normalized_arc_logits = attended_arcs.transpose(1, 2) 
            z = log_partition(attended_arcs, mask)
            batch_size, seq_len = mask.size()
            z = z.unsqueeze(-1).unsqueeze(-1).expand(-1, seq_len, seq_len)
            z = z / (mask.float().sum(dim=1) - 1).view(batch_size, 1, 1)
            z = z * mask.float().unsqueeze(1) * mask.float().unsqueeze(2)
            normalized_arc_logits = normalized_arc_logits - z

        return normalized_arc_logits, normalized_pairwise_head_logits, lengths

    def get_arc_factored_probs(
        self,
        embedded_text_input: torch.Tensor,
        pos_tags: torch.LongTensor,
        mask: torch.Tensor,
        head_tags: torch.LongTensor = None,
        head_indices: torch.LongTensor = None,
        langs: torch.LongTensor = None,
        variational: bool = False,
    ):
        if self.VIB is not None: 
            bottlenecked_text, head_indices, head_tags, pos_tags, mask, kl_loss, kl_div, kl_div2 = \
                self._bottleneck(embedded_text_input, pos_tags, mask, head_tags,
                                 head_indices, langs, variational)
            embedded_text = bottlenecked_text
        else:
            embedded_text = embedded_text_input

        if self.typo_encoder is not None and self.typo_feedforward is not None:
            expanded_feats = self._gen_typo_feats(embedded_text_input, mask)
            augmented_text = torch.cat([embedded_text, expanded_feats], dim=-1)
            encoded_text = self.encoder(augmented_text, mask)
        else:
            encoded_text = self.encoder(embedded_text, mask)

        head_tag_representation, child_tag_representation, attended_arcs, mask, \
            head_tags, head_indices = self._project(encoded_text, mask, head_tags, head_indices)
        normalized_arc_logits, normalized_pairwise_head_logits, lengths = self._attend_and_normalize(
            head_tag_representation, child_tag_representation, attended_arcs, mask)

        return {"arc_log_probs": normalized_arc_logits,
                "tag_log_probs": normalized_pairwise_head_logits,
                "lengths": lengths,
                "mask": mask}

    @overrides
    def forward(
        self,  # type: ignore
        words: TextFieldTensors,
        pos_tags: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        head_tags: torch.LongTensor = None,
        head_indices: torch.LongTensor = None,
        lemmas: TextFieldTensors = None,
        feats: TextFieldTensors = None,
        langs: torch.LongTensor = None,
        return_metric: bool = False,
        variational: bool = False,
        loss_ratios: Dict[str, float] = {"dep": 1.0, "pos": 0.0},
    ) -> Dict[str, torch.Tensor]:

        """
        Embedding each language by the corresponding parameters for
        `TextFieldEmbedder`. Batches should contain only samples from a
        single language.
        Metadata should have a `lang` key.
        """
        if "lang" not in metadata[0]:
            raise ConfigurationError(
                "metadata is missing 'lang' key; "
                "Use the universal_dependencies_multilang dataset_reader."
            )

        batch_lang = metadata[0]["lang"]
        for entry in metadata:
            if entry["lang"] != batch_lang:
                raise ConfigurationError("Two languages in the same batch.")

        metric = {}

        mask = get_text_field_mask(words)
        embedded_text_input = self._embed(words, pos_tags, mask, metadata, lemmas, feats, langs)

        kl_loss = None
        pos_loss = None
        if self.VIB is not None: 
            bottlenecked_text, head_indices, head_tags, pos_tags, mask, kl_loss, kl_div, kl_div2 = \
                self._bottleneck(embedded_text_input, pos_tags, mask, head_tags,
                                 head_indices, langs, variational)
            embedded_text = bottlenecked_text
            kl_div_score = self._lang_kl_divs[batch_lang]
            kl_div_score(kl_div)
            if loss_ratios["pos"] > 0.0:
                logits = self.tag_projection_layer(bottlenecked_text)
                pos_loss = sequence_cross_entropy_with_logits(logits, pos_tags, mask)
                for name, pos_metric in self._pos_metrics.items():
                    pos_metric(logits, pos_tags, mask.float())
                    metric.update({name: pos_metric.get_metric(reset=True)})
                if loss_ratios["dep"] == 0.0:
                    return {
                        "loss": pos_loss,
                        "metric": metric
                    }
        else:
            embedded_text = embedded_text_input

        if self.typo_encoder is not None and self.typo_feedforward is not None:
            expanded_feats = self._gen_typo_feats(embedded_text_input, mask)
            augmented_text = torch.cat([embedded_text, expanded_feats], dim=-1)
            encoded_text = self.encoder(augmented_text, mask)
        else:
            encoded_text = self.encoder(embedded_text, mask)

        if variational:
            return {"kl_loss": kl_loss, "kl_div": kl_div, "kl_div2": kl_div2}

        head_arc_representation, child_arc_representation, \
        head_tag_representation, child_tag_representation, attended_arcs, mask, \
            head_tags, head_indices = self._project(encoded_text, mask, head_tags, head_indices,
                                                    return_arc_representation=True)
        predicted_heads, predicted_head_tags, mask, arc_nll, tag_nll, per_sample_loss = \
        self._parse(
            head_tag_representation, child_tag_representation,
            attended_arcs, mask, head_tags, head_indices,
        )

        loss = loss_ratios["dep"] * (arc_nll + tag_nll)
        if loss_ratios["pos"] > 0.0:
            loss = loss + loss_ratios["pos"] * pos_loss

        if head_indices is not None and head_tags is not None:
            evaluation_mask = self._get_mask_for_eval(mask[:, 1:], pos_tags)
            # We calculate attatchment scores for the whole sentence
            # but excluding the symbolic ROOT token at the start,
            # which is why we start from the second element in the sequence.
            scores = self._lang_attachment_scores[batch_lang]
            scores(
                predicted_heads[:, 1:],
                predicted_head_tags[:, 1:],
                head_indices[:, 1:],
                head_tags[:, 1:],
                evaluation_mask,
            )
            if return_metric:
                metric.update(scores.get_metric(reset=True))
                if self.VIB:
                    kl_div_metric = kl_div_score.get_metric(reset=True)
                    metric.update({"kl_div": kl_div_metric})

        output_dict = {}
        if self._inspect_layer == "embedding":
            output_dict["hidden_state"] = embedded_text_input
        elif self._inspect_layer == "vib":
            output_dict["hidden_state"] = bottlenecked_text
        elif self._inspect_layer == "encoder":
            output_dict["hidden_state"] = encoded_text
        elif self._inspect_layer == "projection":
            batch_size, sequence_length, _ = head_arc_representation.size()
            range_vector = get_range_vector(batch_size, get_device_of(head_arc_representation)).unsqueeze(1)
            timestep_index = get_range_vector(sequence_length, get_device_of(head_arc_representation))
            child_index = (
                timestep_index.view(1, sequence_length).expand(batch_size, sequence_length).long()
            )
            # we're exchanging child and head arc since this:
            # https://github.com/allenai/allennlp/issues/2908
            head_arc_reps = child_arc_representation[range_vector, head_indices]
            child_arc_reps = head_arc_representation[range_vector, child_index]
            head_tag_reps = head_tag_representation[range_vector, head_indices]
            child_tag_reps = child_tag_representation[range_vector, child_index]
            output_dict["arc_hidden_state"] = torch.cat([head_arc_reps, child_arc_reps], dim=-1)
            output_dict["tag_hidden_state"] = torch.cat([head_tag_reps, child_tag_reps], dim=-1)
        else:
            raise NotImplementedError

        output_dict.update({
            "heads": predicted_heads,
            "head_tags": predicted_head_tags,
            "arc_loss": arc_nll,
            "tag_loss": tag_nll,
            "loss": loss,
            "mask": mask,
            "per_sample_loss": per_sample_loss,
        })

        if metric:
            output_dict["metric"] = metric

        if kl_loss is not None:
            output_dict["kl_loss"] = kl_loss

        if pos_loss is not None:
            output_dict["pos_loss"] = pos_loss

        self._add_metadata_to_output_dict(metadata, output_dict)

        return output_dict

    @staticmethod
    def _add_metadata_to_output_dict(metadata, output_dict):
        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]
            output_dict["langs"] = [x["lang"] for x in metadata]
            output_dict["upos"] = [x["upos"] for x in metadata]
            output_dict["xpos"] = [x["xpos"] for x in metadata]
            output_dict["feats"] = [x["feats"] for x in metadata]
            output_dict["lemmas"] = [x["lemmas"] for x in metadata]
            output_dict["gold_tags"] = [x["gold_tags"] for x in metadata]
            output_dict["gold_heads"] = [x["gold_heads"] for x in metadata]
            output_dict["ids"] = [x["ids"] for x in metadata if "ids" in x]
            output_dict["multiword_ids"] = [x["multiword_ids"] for x in metadata if "multiword_ids" in x]
            output_dict["multiword_forms"] = [x["multiword_forms"] for x in metadata if "multiword_forms" in x]

        return output_dict

    def _apply_token_dropout(self, words, mask, dropout, offsets = None, form = 'subword'):
        # Word dropout

        def mask_words(tokens, drop_mask, drop_token):
            drop_fill = tokens.new_empty(tokens.size()).long().fill_(drop_token)
            return torch.where(drop_mask, drop_fill, tokens)

        if form == "tensor":
            assert isinstance(words, torch.Tensor)
            drop_mask = self._get_dropout_mask(mask.bool(),
                                               p=dropout,
                                               training=self.training)
            mean = masked_mean(words, mask=mask.unsqueeze(-1), dim=(0,1), keepdim=True)
            expanded_mean = mean.expand_as(words)
            words = torch.where(drop_mask.unsqueeze(-1), expanded_mean, words)

        if form == "word":
            drop_mask = self._get_dropout_mask(mask.bool(),
                                               p=dropout,
                                               training=self.training)
            drop_token = self.vocab.get_token_index(self.vocab._oov_token)
            words = mask_words(words, drop_mask, drop_token)

        def mask_subwords(token_ids, offsets, drop_mask, drop_token):
            subword_drop_mask = token_ids.new_zeros(token_ids.size()).bool()
            batch_size, seq_len, _ = offsets.size()
            for i in range(batch_size):
                for j in range(seq_len):
                    start, end = offsets[i,j].tolist()
                    subword_drop_mask[i, start:(end+1)] = drop_mask[i,j]
            drop_fill = token_ids.new_empty(token_ids.size()).long().fill_(drop_token)
            return torch.where(subword_drop_mask, drop_fill, token_ids)

        if form == "subword":
            assert offsets is not None
            drop_mask = self._get_dropout_mask(mask.bool(),
                                               p=dropout,
                                               training=self.training)
            drop_token = self._tokenizer.encode("[MASK]", add_special_tokens=False)[0]
            words = mask_subwords(words, offsets, drop_mask, drop_token)

        return words

    @staticmethod
    def _get_dropout_mask(mask: torch.Tensor,
                          p: float = 0.0,
                          training: float = True) -> torch.LongTensor:
        """
        During training, randomly replaces some of the non-padding tokens to a mask token with probability ``p``

        :param tokens: The current batch of padded sentences with word ids
        :param drop_token: The mask token
        :param padding_tokens: The tokens for padding the input batch
        :param p: The probability a word gets mapped to the unknown token
        :param training: Applies the dropout if set to ``True``
        :return: A copy of the input batch with token dropout applied
        """
        if training and p > 0:
            # Create a uniformly random mask selecting either() the original words or OOV tokens
            dropout_mask = (mask.new_empty(mask.size()).float().uniform_() < p)
            drop_mask = dropout_mask & mask
            return drop_mask
        else:
            return mask.new_zeros(mask.size()).bool()

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        head_tags = output_dict.pop("head_tags").cpu().detach().numpy()
        heads = output_dict.pop("heads").cpu().detach().numpy()
        mask = output_dict.pop("mask")
        lengths = get_lengths_from_binary_sequence_mask(mask)
        head_tag_labels = []
        head_indices = []
        for instance_heads, instance_tags, length in zip(heads, head_tags, lengths):
            instance_heads = list(instance_heads[1:length])
            instance_tags = instance_tags[1:length]
            labels = [
                self.vocab.get_token_from_index(label, "head_tags") for label in instance_tags
            ]
            head_tag_labels.append(labels)
            head_indices.append(instance_heads)

        output_dict["predicted_dependencies"] = head_tag_labels
        output_dict["predicted_heads"] = head_indices
        return output_dict

    def _parse(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        attended_arcs: torch.Tensor,
        mask: torch.LongTensor,
        head_tags: torch.LongTensor = None,
        head_indices: torch.LongTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.training or not self.use_mst_decoding_for_validation:
            predicted_heads, predicted_head_tags = self._greedy_decode(
                head_tag_representation, child_tag_representation, attended_arcs, mask
            )
        else:
            predicted_heads, predicted_head_tags = self._mst_decode(
                head_tag_representation, child_tag_representation, attended_arcs, mask
            )
        if head_indices is not None and head_tags is not None:

            arc_nll, tag_nll, per_sample_loss = self._construct_loss(
                head_tag_representation=head_tag_representation,
                child_tag_representation=child_tag_representation,
                attended_arcs=attended_arcs,
                head_indices=head_indices,
                head_tags=head_tags,
                mask=mask,
            )
        else:
            arc_nll, tag_nll, per_sample_loss = self._construct_loss(
                head_tag_representation=head_tag_representation,
                child_tag_representation=child_tag_representation,
                attended_arcs=attended_arcs,
                head_indices=predicted_heads.long(),
                head_tags=predicted_head_tags.long(),
                mask=mask,
            )

        return predicted_heads, predicted_head_tags, mask, arc_nll, tag_nll, per_sample_loss

    def _construct_loss(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        attended_arcs: torch.Tensor,
        head_indices: torch.Tensor,
        head_tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.

        # Parameters

        head_tag_representation : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        head_indices : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length).
            The indices of the heads for every word.
        head_tags : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length).
            The dependency labels of the heads for every word.
        mask : `torch.Tensor`, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        # Returns

        arc_nll : `torch.Tensor`, required.
            The negative log likelihood from the arc loss.
        tag_nll : `torch.Tensor`, required.
            The negative log likelihood from the arc tag loss.
        """
        float_mask = mask.float()
        batch_size, sequence_length, _ = attended_arcs.size()
        range_vector = get_range_vector(batch_size, get_device_of(attended_arcs)).unsqueeze(1)
        timestep_index = get_range_vector(sequence_length, get_device_of(attended_arcs))
        child_index = (
            timestep_index.view(1, sequence_length).expand(batch_size, sequence_length).long()
        )
        if self.use_crf:
            z = log_partition(attended_arcs, mask)
            attended_arcs = attended_arcs * float_mask.unsqueeze(2) * float_mask.unsqueeze(1)
            arc_loss = attended_arcs[range_vector, child_index, head_indices]
            arc_loss = arc_loss[:, 1:].sum(dim=1) - z
            if torch.any(arc_loss != arc_loss):
                logger.warning(f"nan found: {torch.nonzero(arc_loss != arc_loss)}")
            arc_loss[arc_loss != arc_loss] = 0
            mask = mask * (arc_loss == arc_loss).float().unsqueeze(-1)
            float_mask = mask.float()
        else:
            normalised_arc_logits = (
                masked_log_softmax(attended_arcs, mask)
                * float_mask.unsqueeze(2)
                * float_mask.unsqueeze(1)
            )
            arc_loss = normalised_arc_logits[range_vector, child_index, head_indices]
            arc_loss = arc_loss[:, 1:]

        head_tag_logits = self._get_head_tags(
            head_tag_representation, child_tag_representation, head_indices
        )
        normalised_head_tag_logits = masked_log_softmax(
            head_tag_logits, mask.unsqueeze(-1)
        ) * float_mask.unsqueeze(-1)

        tag_loss = normalised_head_tag_logits[range_vector, child_index, head_tags]
        tag_loss = tag_loss[:, 1:]

        # The number of valid positions is equal to the number of unmasked elements minus
        # 1 per sequence in the batch, to account for the symbolic HEAD token.
        valid_positions = mask.sum() - batch_size

        arc_nll = -arc_loss.sum() / valid_positions.float()
        tag_nll = -tag_loss.sum() / valid_positions.float()

        if self.use_crf:
            arc_sample_nll = -arc_loss / (mask.sum(dim=1) - 1).float()
        else:
            arc_sample_nll = -arc_loss.sum(dim=1) / (mask.sum(dim=1) - 1).float()

        tag_sample_nll = -tag_loss.sum(dim=1) / (mask.sum(dim=1) - 1).float()

        per_sample_loss = arc_sample_nll + tag_sample_nll
        return arc_nll, tag_nll, per_sample_loss

    def _greedy_decode(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        attended_arcs: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions by decoding the unlabeled arcs
        independently for each word and then again, predicting the head tags of
        these greedily chosen arcs independently. Note that this method of decoding
        is not guaranteed to produce trees (i.e. there maybe be multiple roots,
        or cycles when children are attached to their parents).

        # Parameters

        head_tag_representation : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.

        # Returns

        heads : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the greedily decoded heads of each word.
        """
        # Mask the diagonal, because the head of a word can't be itself.
        attended_arcs = attended_arcs + torch.diag(
            attended_arcs.new(mask.size(1)).fill_(-numpy.inf)
        )
        # Mask padded tokens, because we only want to consider actual words as heads.
        if mask is not None:
            minus_mask = (1 - mask).to(dtype=torch.bool).unsqueeze(2)
            attended_arcs.masked_fill_(minus_mask, -numpy.inf)

        # Compute the heads greedily.
        # shape (batch_size, sequence_length)
        _, heads = attended_arcs.max(dim=2)

        # Given the greedily predicted heads, decode their dependency tags.
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(
            head_tag_representation, child_tag_representation, heads
        )
        _, head_tags = head_tag_logits.max(dim=2)
        return heads, head_tags

    def _mst_decode(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        attended_arcs: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions using the Edmonds' Algorithm
        for finding minimum spanning trees on directed graphs. Nodes in the
        graph are the words in the sentence, and between each pair of nodes,
        there is an edge in each direction, where the weight of the edge corresponds
        to the most likely dependency label probability for that arc. The MST is
        then generated from this directed graph.

        # Parameters

        head_tag_representation : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.

        # Returns

        heads : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the optimally decoded heads of each word.
        """

        normalized_arc_logits, normalized_pairwise_head_logits, lengths = self._attend_and_normalize(
            head_tag_representation,
            child_tag_representation,
            attended_arcs,
            mask,
        )
        # Shape (batch_size, num_head_tags, sequence_length, sequence_length)
        # This energy tensor expresses the following relation:
        # energy[i,j] = "Score that i is the head of j". In this
        # case, we have heads pointing to their children.
        batch_energy = torch.exp(
            normalized_arc_logits.unsqueeze(1) + normalized_pairwise_head_logits
        )
        return self._run_mst_decoding(batch_energy, lengths)

    @staticmethod
    def _run_mst_decoding(
        batch_energy: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        head_tags = []
        for energy, length in zip(batch_energy.detach().cpu(), lengths):
            scores, tag_ids = energy.max(dim=0)
            # Although we need to include the root node so that the MST includes it,
            # we do not want any word to be the parent of the root node.
            # Here, we enforce this by setting the scores for all word -> ROOT edges
            # edges to be 0.
            scores[0, :] = 0
            # Decode the heads. Because we modify the scores to prevent
            # adding in word -> ROOT edges, we need to find the labels ourselves.
            instance_heads, _ = decode_mst(scores.numpy(), length, has_labels=False)

            # Find the labels which correspond to the edges in the max spanning tree.
            instance_head_tags = []
            for child, parent in enumerate(instance_heads):
                instance_head_tags.append(tag_ids[parent, child].item())
            # We don't care what the head or tag is for the root token, but by default it's
            # not necesarily the same in the batched vs unbatched case, which is annoying.
            # Here we'll just set them to zero.
            instance_heads[0] = 0
            instance_head_tags[0] = 0
            heads.append(instance_heads)
            head_tags.append(instance_head_tags)
        return torch.from_numpy(numpy.stack(heads)), torch.from_numpy(numpy.stack(head_tags))

    def _get_head_tags(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        head_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decodes the head tags given the head and child tag representations
        and a tensor of head indices to compute tags for. Note that these are
        either gold or predicted heads, depending on whether this function is
        being called to compute the loss, or if it's being called during inference.

        # Parameters

        head_tag_representation : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        head_indices : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length). The indices of the heads
            for every word.

        # Returns

        head_tag_logits : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length, num_head_tags),
            representing logits for predicting a distribution over tags
            for each arc.
        """
        batch_size = head_tag_representation.size(0)
        # shape (batch_size,)
        range_vector = get_range_vector(
            batch_size, get_device_of(head_tag_representation)
        ).unsqueeze(1)

        # This next statement is quite a complex piece of indexing, which you really
        # need to read the docs to understand. See here:
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing
        # In effect, we are selecting the indices corresponding to the heads of each word from the
        # sequence length dimension for each element in the batch.

        # shape (batch_size, sequence_length, tag_representation_dim)
        selected_head_tag_representations = head_tag_representation[range_vector, head_indices]
        selected_head_tag_representations = selected_head_tag_representations.contiguous()
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self.tag_bilinear(
            selected_head_tag_representations, child_tag_representation
        )
        return head_tag_logits

    def _get_mask_for_eval(
        self, mask: torch.LongTensor, pos_tags: torch.LongTensor
    ) -> torch.LongTensor:
        """
        Dependency evaluation excludes words are punctuation.
        Here, we create a new mask to exclude word indices which
        have a "punctuation-like" part of speech tag.

        # Parameters

        mask : `torch.LongTensor`, required.
            The original mask.
        pos_tags : `torch.LongTensor`, required.
            The pos tags for the sequence.

        # Returns

        A new mask, where any indices equal to labels
        we should be ignoring are masked.
        """
        new_mask = mask.detach()
        for label in self._pos_to_ignore:
            label_mask = pos_tags.eq(label).long()
            new_mask = new_mask * (1 - label_mask)
        return new_mask

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        all_uas = []
        all_las = []
        for lang, scores in self._lang_attachment_scores.items():
            lang_metrics = scores.get_metric(reset)

            for key in lang_metrics.keys():
                # Store only those metrics.
                if key in ["UAS", "LAS", "loss"]:
                    metrics["{}_{}".format(key, lang)] = lang_metrics[key]

            # Include in the average only languages that should count for early stopping.
            #if lang in self._langs_for_early_stop:
            all_uas.append(metrics["UAS_{}".format(lang)])
            all_las.append(metrics["LAS_{}".format(lang)])

        #if self._langs_for_early_stop:
        metrics.update({"UAS_AVG": numpy.mean(all_uas), "LAS_AVG": numpy.mean(all_las)})

        return metrics

    def add_ft_lang_mean_to_lang_means(self, ft_lang_mean):
        self._ft_lang_mean = torch.nn.Parameter(ft_lang_mean, requires_grad=False)

