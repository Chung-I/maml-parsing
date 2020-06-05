from typing import Dict, Optional, Any, List, Tuple
import logging
from functools import partial
import math
import numpy as np
from overrides import overrides
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_ as he_init

from src.training.util import INF, get_dir_mask, avg_loss_func
from src.modules.dmv import DMV
from src.modules.projection import NICETrans
from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding, InputVariationalDropout
from allennlp.modules import FeedForward, TimeDistributed, Seq2VecEncoder
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
from src.training.metrics import AttachmentScores

from src.training.util import get_lang_means, get_lang_mean

logger = logging.getLogger(__name__)


class ResidualLayer(nn.Module):
    def __init__(self, in_dim=100,
                 out_dim=100):
        super(ResidualLayer, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return F.relu(self.lin2(F.relu(self.lin1(x)))) + x


@Model.register("neuraldmv")
class NeuralLDMVModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        dmv: DMV,
        nice_layer: NICETrans,
        encoder: Seq2SeqEncoder,
        hidden_dim: int,
        state_dim: int,
        n_states: int,
        max_len: int = 30,
        lang_mean_regex: str = None,
        function_words: List[str] = None,
        average: str = 'batch',
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):

        super(NeuralLDMVModel, self).__init__(vocab, regularizer)

        self._lang_attachment_scores: Dict[str, AttachmentScores] = defaultdict(AttachmentScores)
        # temporary hyper-param for training or validation
        self.to_parse = False

        self.text_field_embedder = text_field_embedder
        self.input_dim = text_field_embedder.get_output_dim()
        num_langs = self.vocab.get_vocab_size("lang_labels")
        self.dmv = dmv
        self.n_vals = dmv.valency_num
        self.n_cvals = dmv.cvalency_num
        self.n_dirs = dmv.dir_num
        self.hidden_dim = hidden_dim
        self.n_cls = 2
        self.n_states = n_states
        self.state_dim = state_dim
        self._pad_idx = self.vocab.get_token_index(self.vocab._padding_token)

        self.encoder = encoder
        self.enc_linear = nn.Linear(self.encoder.get_output_dim(), hidden_dim*2)

        self.masked_indices = None
        if function_words is not None:
            self.masked_indices = [self.vocab.get_token_index(word, namespace="pos")
                                   for word in function_words]

        self.state_emb = nn.Parameter(torch.Tensor(n_states, state_dim))
        self.left_mlp = nn.Sequential(
            nn.Linear((state_dim + hidden_dim), state_dim),
            ResidualLayer(state_dim, state_dim),
            ResidualLayer(state_dim, state_dim),
            nn.Linear(state_dim, self.n_states),
        )
        self.right_mlp = nn.Sequential(
            nn.Linear((state_dim + hidden_dim), state_dim),
            ResidualLayer(state_dim, state_dim),
            ResidualLayer(state_dim, state_dim),
            nn.Linear(state_dim, self.n_states),
        )

        self.stop_left = nn.Sequential(
            nn.Linear((state_dim + hidden_dim), state_dim),
            ResidualLayer(state_dim, state_dim),
            ResidualLayer(state_dim, state_dim),
            nn.Linear(state_dim, self.n_vals * self.n_cls),
        )

        self.stop_right = nn.Sequential(
            nn.Linear((state_dim + hidden_dim), state_dim),
            ResidualLayer(state_dim, state_dim),
            ResidualLayer(state_dim, state_dim),
            nn.Linear(state_dim, self.n_vals * self.n_cls),
        )
        self.root_attach_left = nn.Sequential(
            nn.Linear((state_dim + hidden_dim), state_dim),
            ResidualLayer(state_dim, state_dim),
            ResidualLayer(state_dim, state_dim),
            nn.Linear(state_dim, 1),
        )
        self.nice_layer = nice_layer
        self._lang_means = None
        if lang_mean_regex is not None:
            lang_means = get_lang_means(lang_mean_regex, self.vocab)
            self._lang_means = torch.nn.Parameter(lang_means, requires_grad=False)
        self.r_mean = torch.nn.Parameter(torch.Tensor(self.n_states, self.input_dim))
        self.r_std = torch.nn.Parameter(torch.Tensor(self.n_states, self.input_dim))
        self.loss_func = partial(avg_loss_func, average=average)
        initializer(self)

    @property
    def r_var(self):
        return (self.r_std ** 2) ** 0.5

    def get_mask(self, tensor):
        transition_mask = tensor.new_ones(
            self.n_states, self.n_states).bool()
        transition_mask[:, ROOT_ID] = False
        func_word_mask = tensor.new_ones(
            self.n_states, self.n_states).bool()
        #if self.masked_indices is not None:
        #    func_word_mask[self.masked_indices, :] = False

        right_decision_mask = tensor.new_ones(
            self.n_vals, self.n_states, self.n_cls).bool()
        right_decision_mask[0, ROOT_ID, 0] = False
        right_decision_mask[1, ROOT_ID, 1] = False

        return transition_mask, right_decision_mask, func_word_mask

    def kl(self, mean, logvar):
        result =  -0.5 * (logvar - torch.pow(mean, 2)- torch.exp(logvar) + 1)
        return result

    def _normalize(self, embedded_text_input, langs):

        if self._lang_means is not None:
            batch_size, seq_len, _ = embedded_text_input.size()
            if langs is None:
                means = self._ft_lang_mean.view(1, 1, -1).repeat(batch_size, seq_len, 1)
            else:
                expanded_langs = langs.unsqueeze(-1).repeat(1, seq_len)
                means = self._lang_means[expanded_langs] 
            embedded_text_input = embedded_text_input - means

        return embedded_text_input

    def _eval_log_density(self, s, mask):
        """
        Args:
            s: A tensor with size (batch_size, seq_length, features)

        Returns:
            density: (batch_size, seq_length, num_state)

        """
        log_density_c = self._calc_log_density_c()
        batch_size, seq_len, feat_dim = s.size()
        ep_size = torch.Size([batch_size, seq_len, self.n_states, feat_dim])
        means = self.r_mean.view(1, 1, self.n_states, feat_dim).expand(ep_size)
        means = s.new_zeros(ep_size)
        words = s.unsqueeze(dim=2).expand(ep_size)
        mask = mask.view(batch_size, seq_len, 1, 1).expand(ep_size)
        var = self.r_var.expand(ep_size)
        tmp = 0.5 * torch.sum(mask * (means - words) ** 2 / var, dim=3)
        return log_density_c.view(1, 1, self.n_states) - tmp


    def _calc_log_density_c(self):

        return -self.input_dim/2.0 * (math.log(2 * math.pi)) - \
                0.5 * torch.sum(torch.log(self.r_var), dim=1)

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
        use_mean: bool = False,
    ) -> Dict[str, torch.Tensor]:

        output_dict: Dict[str, Any] = {}
        lang_id = langs[0]
        batch_lang = metadata[0]["lang"]
        assert torch.all(langs == langs[0].item())
        mask = get_text_field_mask(words)

        embedded_text_input = self.text_field_embedder(words, lang=batch_lang)
        batch_size, seq_len, _ = embedded_text_input.size()
        embedded_text_input = self._normalize(embedded_text_input, langs)
        encoded_text = self.encoder(embedded_text_input, mask)
        params = self.enc_linear(encoded_text.max(1)[0])
        mean = params[:, :self.hidden_dim]
        logvar = params[:, self.hidden_dim:]
        kl = self.kl(mean, logvar).sum(1)
        if use_mean:
            z = mean
        else:
            eps = mean.new(batch_size, mean.size(1)).normal_(0, 1)
            z = (0.5*logvar).exp() * eps + mean

        state_emb = self.state_emb
        state_emb = state_emb.view(1, self.n_states, self.state_dim).expand(batch_size, -1, -1)
        z_expand = z.view(batch_size, 1, self.hidden_dim).expand(-1, self.n_states, -1)
        state_emb = torch.cat([state_emb, z_expand], dim=-1)

        attach_left_scores = F.log_softmax(self.left_mlp(state_emb), dim=-1).unsqueeze(1).expand(-1, seq_len, -1, -1)
        attach_right_scores = F.log_softmax(self.right_mlp(state_emb), dim=-1).unsqueeze(1).expand(-1, seq_len, -1, -1)
        stop_left_scores = F.log_softmax(self.stop_left(state_emb).view(batch_size, self.n_states, 2, 2), dim=-1)\
            .unsqueeze(1).expand(-1, seq_len, -1, -1, -1)
        stop_right_scores = F.log_softmax(self.stop_right(state_emb).view(batch_size, self.n_states, 2, 2), dim=-1)\
            .unsqueeze(1).expand(-1, seq_len, -1, -1, -1)
        decision_scores = torch.stack([stop_left_scores, stop_right_scores], dim=3)
        root_decision_scores = np.array([[[-INF, -INF],
                                          [-INF, -INF]],
                                         [[-INF, 0],
                                          [0, -INF]]], dtype=np.float32)
        root_decision_scores = torch.from_numpy(root_decision_scores).to(decision_scores.device)\
            .view(1, 1, 1, self.n_dirs, self.n_vals, self.n_cls).expand(batch_size, 1, self.n_states, -1, -1, -1)
        decision_scores = torch.cat([root_decision_scores, decision_scores], dim=1)
        root_attach_left_scores = F.log_softmax(self.root_attach_left(state_emb).squeeze(-1), dim=-1)\
            .unsqueeze(-2).expand(-1, self.n_states, -1).unsqueeze(1)
        attach_left_scores = torch.cat([root_attach_left_scores, attach_left_scores], dim=1)
        root_attach_right_scores = attach_right_scores.new_full(root_attach_left_scores.size(), -INF)
        attach_right_scores = torch.cat([root_attach_right_scores, attach_right_scores], dim=1)
        emission_means, _ = self.nice_layer(embedded_text_input)
        unary_scores = self._eval_log_density(emission_means, mask)
        unary_scores = torch.cat([unary_scores.new_zeros(batch_size, 1, self.n_states), unary_scores], 1)
        mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)


        # batch_transition_scores, batch_decision_scores, batch_unary_scores = \
        #     self._get_batch_scores(
        #         attach_left_scores,
        #         attach_right_scores,
        #         stop_left_scores,
        #         stop_right_scores,
        #         root_attach_left_scores,
        #         unary_scores,
        #         mask,
        #     )

        sent_lens = mask.sum(dim=-1).int()
        partition_score = self.dmv._inside(
            attach_left_scores, attach_right_scores, decision_scores,
            unary_scores, sent_lens,
        )
        loss_mask = partition_score != partition_score
        partition_score.masked_fill_(loss_mask, 0.0)
        loss = self.loss_func(-partition_score + kl, sent_lens.masked_fill(loss_mask, 0))
        output_dict["loss"] = loss
        metrics: Dict[str, Any] = {}
        pred_heads, pred_tags, pred_head_valences, pred_valences = self.dmv._viterbi(
            attach_left_scores, attach_right_scores, decision_scores,
            unary_scores, sent_lens,
        )
        scores = self._lang_attachment_scores[batch_lang]
        scores(pred_heads, head_indices, mask[:, 1:])
        if return_metric:
            metrics.update(scores.get_metric(reset=True))
        output_dict["metric"] = metrics

        return output_dict

    def parse(self, pos_tags, lang_id=None, transition_scores=None,
              decision_scores=None, unary_scores=None, head_ids=None, prepend_root=True):
        if transition_scores is None or decision_scores is None or unary_scores is None:
            assert lang_id is not None, "lang_id is required if scores is not provided"
            transition_scores, decision_scores, unary_scores = self.get_rules(lang_id)

        if prepend_root:
            pos_tags = torch.cat(
                (pos_tags.new_full((pos_tags.size(0), 1), ROOT_ID), pos_tags), dim=-1)
        batch_transition_scores, batch_decision_scores, batch_unary_scores = \
            self.get_batch_scores(pos_tags, transition_scores, decision_scores, unary_scores)

        sent_lens = (pos_tags != PAD_ID).sum(dim=-1)
        pred_heads, pred_tags, pred_head_valences, pred_valences = self.dmv._viterbi(
            batch_transition_scores, batch_decision_scores, batch_unary_scores, sent_lens)
        if head_ids is not None:
            mask = head_ids != PAD_ID
            if lang_id not in self.uas:
                self.uas[lang_id] = AttachmentScores()
            self.uas[lang_id](pred_heads, head_ids, mask)

        return pred_heads

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        all_uas = []
        all_uem = []
        for lang, scores in self._lang_attachment_scores.items():
            lang_metrics = scores.get_metric(reset)

            for key in lang_metrics.keys():
                # Store only those metrics.
                if key in ["UAS", "UEM", "loss"]:
                    metrics["{}_{}".format(key, lang)] = lang_metrics[key]

            # Include in the average only languages that should count for early stopping.
            #if lang in self._langs_for_early_stop:
            all_uas.append(metrics["UAS_{}".format(lang)])
            all_uem.append(metrics["UEM_{}".format(lang)])

        #if self._langs_for_early_stop:
        metrics.update({"UAS_AVG": np.mean(all_uas), "UEM_AVG": np.mean(all_uem)})

        return metrics

    def _get_batch_scores(self,
                          attach_left_scores,
                          attach_right_scores,
                          stop_left_scores,
                          stop_right_scores,
                          root_attach_left_scores,
                          unary_scores,
                          mask):
        placeholder = attach_left_scores
        assert self.n_cvals == 1

        n_states = self.n_states
        # add root
        batch_size, seq_len, _ = unary_scores.size()
        batch_transition_scores = placeholder.new_full(
            (batch_size, seq_len-1, seq_len, n_states, n_states), -INF)
        batch_root_attach_left_scores = \
            root_attach_left_scores.view(batch_size, 1, seq_len, 1, n_states).expand(-1, -1, -1, n_states, -1)
        batch_transition_scores = torch.cat([batch_root_attach_left_scores, batch_transition_scores], dim=1)
        root_decision_scores = np.array([[[-INF, -INF],
                                          [-INF, -INF]],
                                         [[-INF, 0],
                                          [0, -INF]]], dtype=np.float32)
        root_decision_scores = torch.from_numpy(root_decision_scores).to(placeholder.device)
        batch_root_decision_scores = root_decision_scores.view(1, 1, self.n_dirs, self.n_vals, 1, self.n_cls)\
            .expand(batch_size, 1, -1, -1, n_states, -1)
        #base_decision_scores = placeholder.new_full(
        #    (batch_size, seq_len - 1, self.n_dirs, self.n_vals, n_states, self.n_cls), -INF)
        #base_decision_scores = torch.cat([batch_root_decision_scores, base_decision_scores], dim=1)
        sent_lens = mask.sum(dim=-1)
        left_mask, right_mask = get_dir_mask(sent_lens, root=0)
        left_mask = left_mask.view(batch_size, seq_len, seq_len, 1, 1).expand_as(batch_transition_scores)
        right_mask = right_mask.view(batch_size, seq_len, seq_len, 1, 1).expand_as(batch_transition_scores)
        expanded_size = batch_transition_scores.size()
        batch_transition_scores = torch.where(
            left_mask,
            attach_left_scores.unsqueeze(-3).expand(expanded_size),
            batch_transition_scores,
        )
        batch_transition_scores = torch.where(
            right_mask,
            attach_right_scores.unsqueeze(-3).expand(expanded_size),
            batch_transition_scores,
        )
        batch_transition_scores = \
            batch_transition_scores.unsqueeze(-1).expand(-1, -1, -1, -1, -1, self.n_cvals)

        batch_decision_scores = torch.stack([stop_left_scores[:, 1:], stop_right_scores[:, 1:]], dim=-4)
        batch_decision_scores = torch.where(mask[:, 1:].view(batch_size, seq_len-1, 1, 1, 1, 1).bool(),
                                            batch_decision_scores,
                                            -INF * torch.ones_like(batch_decision_scores))
        batch_decision_scores = torch.cat([batch_root_decision_scores, batch_decision_scores], dim=1)
        batch_decision_scores = batch_decision_scores.permute(0, 1, 4, 2, 3, 5)

        return batch_transition_scores, batch_decision_scores, unary_scores
