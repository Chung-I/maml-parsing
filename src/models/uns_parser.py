from typing import Dict, Optional, Any, List, Tuple
import logging
from functools import partial

import numpy as np

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
from allennlp.training.metrics import AttachmentScores, Average, CategoricalAccuracy

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


@Model.register("neuralldmv")
class NeuralLDMVModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        dmv: DMV,
        nice_layer: NICETrans,
        encoder: Seq2SeqEncoder,
        val_dim: int,
        hidden_dim: int,
        pre_output_dim: int,
        state_dim: int,
        n_states: int,
        lang_dim: int,
        dir_dim: int = 5,
        max_len: int = 30,
        function_words: List[str] = None,
        average: str = 'batch',
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):

        super(NeuralLDMVModel, self).__init__(vocab, regularizer)

        self.uas: Dict[int, AttachmentScores] = dict()
        # temporary hyper-param for training or validation
        self.to_parse = False

        self.text_field_embedder = text_field_embedder
        self.input_dim = text_field_embedder.get_output_dim()
        lang_num = self.vocab.get_vocab_size("lang_labels")
        self.dmv = dmv
        self.n_valss = dmv.valency_num
        self.n_cvals = dmv.cvalency_num
        self.n_dirs = dmv.dir_num
        self.hidden_dim = hidden_dim
        self.pre_output_dim = pre_output_dim
        self.val_dim = val_dim
        self.lang_dim = lang_dim
        self.dir_dim = dir_dim
        self.cls_num = 2
        self.n_states = n_states
        self.state_dim = state_dim
        self._pad_idx = self.vocab.get_token_index(self.vocab._padding_token)

        self.encoder = encoder
        self.enc_linear = nn.Linear(self.encoder.get_output_dim(), hidden_dim*2)

        self.masked_indices = None
        if function_words is not None:
            self.masked_indices = [self.vocab.get_token_index(word, namespace="pos")
                                   for word in function_words]

        self.state_emb = nn.Parameter(torch.randn(n_states, state_dim))
        self.val_emb = nn.Parameter(torch.randn(self.n_vals, val_dim))
        if lang_dim > 0:
            self.lang_embs = nn.Parameter(torch.randn(lang_num, lang_dim))
            self.register_parameter('lang_embs', self.lang_embs)
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
            nn.Linear(state_dim, 4 * self.n_states),
        )

        self.stop_right = nn.Sequential(
            nn.Linear((state_dim + hidden_dim), state_dim),
            ResidualLayer(state_dim, state_dim),
            ResidualLayer(state_dim, state_dim),
            nn.Linear(state_dim, 4 * self.n_states),
        )
        self.root_attach_left = nn.Sequential(
            nn.Linear((state_dim + hidden_dim), state_dim),
            ResidualLayer(state_dim, state_dim),
            ResidualLayer(state_dim, state_dim),
            nn.Linear(state_dim, self.n_states),
        )
        self.nice_layer = nice_layer
        self.r_mean = torch.nn.Parameter(torch.Tensor(max_len, self.n_states, self.input_dim))
        self.r_var = torch.nn.Parameter(torch.Tensor(max_len, self.n_states, self.input_dim))
        self.loss_func = partial(avg_loss_func, average=average)
        initializer(self)

    def get_mask(self, tensor):
        transition_mask = tensor.new_ones(
            self.n_states, self.n_states).bool()
        transition_mask[:, ROOT_ID] = False
        func_word_mask = tensor.new_ones(
            self.n_states, self.n_states).bool()
        #if self.masked_indices is not None:
        #    func_word_mask[self.masked_indices, :] = False

        right_decision_mask = tensor.new_ones(
            self.n_vals, self.n_states, self.cls_num).bool()
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

    def _eval_log_density(self, s):
        """
        Args:
            s: A tensor with size (batch_size, seq_length, features)

        Returns:
            density: (batch_size, seq_length, num_state)

        """
        log_density_c = self._calc_log_density_c()
        batch_size, seq_len, feat_dim = s.size()
        ep_size = torch.Size([batch_size, seq_len, self.n_states, feat_dim])
        means = self.r_mean.view(1, seq_len, self.n_states, feat_dim).expand(ep_size)
        words = s.unsqueeze(dim=2).expand(ep_size)
        var = self.r_var.expand(ep_size)
        return log_density_c - \
               0.5 * torch.sum((means - words) ** 2 / var, dim=3)

    def _calc_log_density_c(self):

        return -self.input_dim/2.0 * (math.log(2 * math.pi)) - \
                0.5 * torch.sum(torch.log(self.r_var))

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

        import pdb
        pdb.set_trace()
        output_dict: Dict[str, Any] = {}
        lang_id = langs[0]
        assert torch.all(langs == langs[0].item())

        transition_scores, decision_scores, unary_scores = self.get_rules(langs[0])

        embedded_text_input = self.text_field_embedder(words, lang=batch_lang)
        batch_size, seq_len, _ = embedded_text_input.size()
        embedded_text_input = self._normalize(embedded_text_input, langs)
        encoded_text = self.encoder(embedded_text_input)
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
        z_expand = z.unsqueeze(1).expand(batch_size, seq_len, self.hidden_dim)
        z_expand = z_expand.unsqueeze(2).expand(batch_size, seq_len, self.n_states, self.hidden_dim)
        state_emb = torch.cat([state_emb, z_expand], 3)

        left_scores = self.left_mlp(state_emb)
        right_scores = self.right_mlp(state_emb)
        stop_left_scores = self.stop_left(state_emb)
        stop_right_scores = self.stop_right(state_emb)
        root_attach_left_scores = self.root_attach_left(state_emb)
        emission_means, _ = self.nice_layer(embedded_text_input)
        unary_scores = self._eval_log_density(emission_means)

        batch_transition_scores, batch_decision_scores, batch_unary_scores = \
            self.get_batch_scores(words, transition_scores, decision_scores, unary_scores)

        sent_lens = (words != self._pad_idx).sum(dim=-1)
        partition_score = self.dmv._inside(
            batch_transition_scores, batch_decision_scores, batch_unary_scores, sent_lens)
        loss_mask = partition_score <= -INF
        partition_score.masked_fill_(loss_mask, 0.0)
        sent_lens.masked_fill_(loss_mask, 0)
        loss = self.loss_func(-partition_score, sent_lens)
        output_dict["loss"] = loss

        if self.to_parse:
            self.parse(words, lang_id, transition_scores, decision_scores, unary_scores,
                       head_indices, prepend_root=False)

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
            # for i in range(5):
            #     print(f"gold parents: {head_ids[i]}")
            #     print(f"pred parents: {pred_heads[i]}")
        return pred_heads

    def get_rules(self, lang_id):

        if self.lang_dim > 0:
            lang_emb = self.lang_embs[lang_id]
            lang_emb = lang_emb.view(1, -1).expand(self.n_states, -1)

        transition_scores = self.state_embs.new_full((self.n_cvals, self.n_dirs, self.n_states, self.n_states),
                                                   -INF)
        decision_scores = self.state_embs.new_full(
            (self.n_vals, self.n_dirs, self.n_states, self.cls_num), -INF)

        transition_mask, right_decision_mask, func_word_mask = \
            self.get_mask(transition_scores)
        for v in range(self.n_vals):
            val_emb = self.val_embs[v].view(1, -1).expand(self.n_states, -1)
            if self.lang_dim > 0:
                feat_embs = torch.cat((self.state_embs, val_emb, lang_emb), dim=-1)
            else:
                feat_embs = torch.cat((self.state_embs, val_emb), dim=-1)
            left_hidden = self.left_mlp(feat_embs)
            right_hidden = self.right_net(feat_embs)
            if v < self.n_cvals:
                left_transition_logits = self.child_net(left_hidden)\
                    .masked_fill_(~transition_mask, -INF)
                transition_scores[v, 0] = F.log_softmax(
                    left_transition_logits, dim=-1)
                right_transition_logits = self.child_net(right_hidden)\
                    .masked_fill_(~transition_mask, -INF)
                transition_scores[v, 1] = F.log_softmax(
                    right_transition_logits, dim=-1)
                if self.training:
                    transition_scores[v, 0].masked_fill_(~func_word_mask, -INF)
                    transition_scores[v, 1].masked_fill_(~func_word_mask, -INF)
            decision_scores[v, 0] = F.log_softmax(
                self.decision_net(left_hidden), dim=-1)
            right_decision_logits = self.decision_net(right_hidden)\
                .masked_fill_(~right_decision_mask[v], -INF)
            decision_scores[v, 1] = F.log_softmax(
                right_decision_logits, dim=-1)

        unary_scores = F.log_softmax(self.vocab_nets[str(lang_id)](self.state_embs), dim=-1)

        return transition_scores, decision_scores, unary_scores

    def get_batch_scores(self, words, transition_scores,
                         decision_scores, unary_scores):
        placeholder = transition_scores

        n_states = self.n_states
        # add root
        batch_size, sent_len = words.size()
        batch_transition_scores = placeholder.new_full(
            (batch_size, sent_len, sent_len, n_states, n_states, self.n_cvals), -INF)
        batch_decision_scores = placeholder.new_full(
            (batch_size, sent_len, n_states, self.n_dirs, self.n_vals, self.cls_num), -INF)
        sent_lens = (words != PAD_ID).sum(dim=-1)
        left_mask, right_mask = get_dir_mask(sent_lens)
        left_mask = left_mask.view(batch_size, sent_len, sent_len, 1, 1).expand_as(batch_transition_scores[..., 0])
        right_mask = right_mask.view(batch_size, sent_len, sent_len, 1, 1).expand_as(batch_transition_scores[..., 0])

        for v in range(self.n_cvals):
            idx_tuple = (words.unsqueeze(-1), words.unsqueeze(-2))
            batch_transition_scores[..., v] = torch.where(
                left_mask, transition_scores[v, 0].view(1, 1, 1, n_states, n_states)\
                .expand_as(batch_transition_scores[..., v]),
                batch_transition_scores[..., v])
            batch_transition_scores[..., v] = torch.where(
                right_mask, transition_scores[v, 1].view(1, 1, 1, n_states, n_states)\
                .expand_as(batch_transition_scores[..., v]),
                batch_transition_scores[..., v])

        for v in range(self.n_vals):
            batch_decision_scores[..., 0, v, :] = \
                decision_scores[v, 0].view(1, 1, n_states, -1).expand(batch_size, sent_len, -1, -1)
            batch_decision_scores[..., 1, v, :] = \
                decision_scores[v, 1].view(1, 1, n_states, -1).expand(batch_size, sent_len, -1, -1)

        batch_unary_scores = unary_scores.transpose(0, 1)[words, :]
        #words_expand = words.unsqueeze(2)\
        #    .expand(batch_size, -1, self.n_states).unsqueeze(3)
        #batch_unary_scores = torch.gather(unary_scores, 3, words_expand).squeeze(3)

        return batch_transition_scores, batch_decision_scores, batch_unary_scores

    def _get_batch_scores(self,
                          attach_left_scores,
                          attach_right_scores,
                          stop_left_scores,
                          stop_right_scores,
                          root_attach_left_scores,
                          unary_scores,
                          mask):
        placeholder = transition_scores
        assert self.n_cvals == 1

        n_states = self.n_states
        # add root
        batch_size, sent_len, _ = unary_scores.size()
        batch_transition_scores = placeholder.new_full(
            (batch_size, sent_len, sent_len, n_states, n_states, self.n_cvals), -INF)
        batch_decision_scores = placeholder.new_full(
            (batch_size, sent_len, n_states, self.n_dirs, self.n_vals, self.cls_num), -INF)
        sent_lens = mask.float().sum(dim=-1)
        left_mask, right_mask = get_dir_mask(sent_lens)
        left_mask = left_mask.view(batch_size, sent_len, sent_len, 1, 1).expand_as(batch_transition_scores[..., 0])
        right_mask = right_mask.view(batch_size, sent_len, sent_len, 1, 1).expand_as(batch_transition_scores[..., 0])

        batch_transition_scores[..., 0] = torch.where(
            left_mask, transition_scores[0].view(1, 1, 1, n_states, n_states)\
            .expand_as(batch_transition_scores[..., 0]),
            batch_transition_scores[..., v])
        batch_transition_scores[..., 0] = torch.where(
            right_mask, transition_scores[1].view(1, 1, 1, n_states, n_states)\
            .expand_as(batch_transition_scores[..., 0]),
            batch_transition_scores[..., 0])

        for v in range(self.n_vals):
            batch_decision_scores[..., 0, v, :] = \
                decision_scores[v, 0].view(1, 1, n_states, -1).expand(batch_size, sent_len, -1, -1)
            batch_decision_scores[..., 1, v, :] = \
                decision_scores[v, 1].view(1, 1, n_states, -1).expand(batch_size, sent_len, -1, -1)

        batch_unary_scores = unary_scores.transpose(0, 1)[words, :]

        return batch_transition_scores, batch_decision_scores, batch_unary_scores
