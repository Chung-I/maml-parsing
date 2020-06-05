import torch
import torch.nn as nn

from src.training.util import constituent_index, INF, _divmod
from allennlp.common import FromParams

class DMV(nn.Module, FromParams):
    def __init__(self,
                 cvalency_num: int,
                 valency_num: int,
                 dir_num: int = 2):
        super(DMV, self).__init__()
        self.cvalency_num = cvalency_num
        self.valency_num = valency_num
        self.dir_num = dir_num

    def _inside(self, left_score, right_score, batch_decision_score, batch_unary_score, sent_lens):
        valency_num = self.valency_num
        cvalency_num = self.cvalency_num
        tensor = left_score
        batch_size, sentence_length, tag_num, _ = left_score.shape
        inside_complete_table = tensor.new_full((batch_size,
                                                 sentence_length * sentence_length * 2,
                                                 tag_num, valency_num), -INF)
        inside_incomplete_table = tensor.new_full((batch_size,
                                                   sentence_length * sentence_length * 2,
                                                   tag_num, tag_num, valency_num), -INF)
        span_2_id, id_2_span, ijss, ikcs, ikis, kjcs, kjis, basic_span = \
            constituent_index(sentence_length, False)

        for ii in basic_span:
            (i, i, dir) = id_2_span[ii]
            inside_complete_table[:, ii, :, :] = batch_decision_score[:, i, :, dir, :, 0]
            inside_complete_table[:, ii, :, :] += batch_unary_score[:, i, :].unsqueeze(-1)

        for ij in ijss:
            (l, r, dir) = id_2_span[ij]
            # two complete span to form an incomplete span
            num_ki = len(ikis[ij])
            inside_ik_ci = inside_complete_table[:, ikis[ij], :, :].reshape(batch_size, num_ki, tag_num, 1, valency_num)
            inside_kj_ci = inside_complete_table[:, kjis[ij], :, :].reshape(batch_size, num_ki, 1, tag_num, valency_num)
            if dir == 0:
                span_inside_i = inside_ik_ci[:, :, :, :, 0].reshape(batch_size, num_ki, tag_num, 1, 1) \
                                + inside_kj_ci[:, :, :, :, 1].reshape(batch_size, num_ki, 1, tag_num, 1) \
                                + right_score[:, r].reshape(batch_size, 1, tag_num, tag_num,
                                                      cvalency_num) \
                                + batch_decision_score[:, r, :, dir, :, 1].reshape(batch_size, 1, 1, tag_num, valency_num)

                # swap head-child to left-right position
            else:
                span_inside_i = inside_ik_ci[:, :, :, :, 1].reshape(batch_size, num_ki, tag_num, 1, 1) \
                                + inside_kj_ci[:, :, :, :, 0].reshape(batch_size, num_ki, 1, tag_num, 1) \
                                + left_score[:, l].reshape(batch_size, 1, tag_num, tag_num, cvalency_num) \
                                + batch_decision_score[:, l, :, dir, :, 1].reshape(batch_size, 1, tag_num, 1, valency_num)

            inside_incomplete_table[:, ij, :, :, :] = torch.logsumexp(span_inside_i, dim=1)

            # one complete span and one incomplete span to form bigger complete span
            num_kc = len(ikcs[ij])
            if dir == 0:
                inside_ik_cc = inside_complete_table[:, ikcs[ij], :, :].reshape(batch_size, num_kc, tag_num, 1, valency_num)
                inside_kj_ic = inside_incomplete_table[:, kjcs[ij], :, :, :].reshape(batch_size, num_kc, tag_num, tag_num,
                                                                                     valency_num)
                span_inside_c = inside_ik_cc[:, :, :, :, 0].reshape(batch_size, num_kc, tag_num, 1, 1) + inside_kj_ic
                span_inside_c = span_inside_c.reshape(batch_size, num_kc * tag_num, tag_num, valency_num)
                inside_complete_table[:, ij, :, :] = torch.logsumexp(span_inside_c, dim=1)
            else:
                inside_ik_ic = inside_incomplete_table[:, ikcs[ij], :, :, :].reshape(batch_size, num_kc, tag_num, tag_num,
                                                                                     valency_num)
                inside_kj_cc = inside_complete_table[:, kjcs[ij], :, :].reshape(batch_size, num_kc, 1, tag_num, valency_num)
                span_inside_c = inside_ik_ic + inside_kj_cc[:, :, :, :, 0].reshape(batch_size, num_kc, 1, tag_num, 1)
                span_inside_c = span_inside_c.transpose(3, 2).reshape(batch_size, num_kc * tag_num, tag_num, valency_num)
                # swap the left-right position since the left tags are to be indexed
                inside_complete_table[:, ij, :, :] = torch.logsumexp(span_inside_c, dim=1)

        final_id = span_2_id[(0, sentence_length - 1, 1)]
        final_ids = [span_2_id[(0, sent_len - 1, 1)] for sent_len in sent_lens.tolist()]
        partition_score = inside_complete_table[torch.arange(batch_size), final_ids, 0, 0]

        #return inside_complete_table, inside_incomplete_table, partition_score
        return partition_score

    def _viterbi(self, left_score, right_score, batch_decision_score, batch_unary_score, sent_lens):
        valency_num = self.valency_num
        cvalency_num = self.cvalency_num
        tensor = left_score
        batch_size, sentence_length, tag_num, _ = left_score.shape

        # CYK table
        complete_table = tensor.new_full((batch_size,
                                          sentence_length * sentence_length * 2,
                                          tag_num, valency_num), -INF)
        incomplete_table = tensor.new_full((batch_size,
                                            sentence_length * sentence_length * 2,
                                            tag_num, tag_num, valency_num), -INF)
        # backtrack table
        complete_backtrack = tensor.new_full((batch_size,
                                              sentence_length * sentence_length * 2,
                                              tag_num, valency_num), -1).int()
        incomplete_backtrack = tensor.new_full((batch_size,
                                                sentence_length * sentence_length * 2,
                                                tag_num, tag_num, valency_num), -1).int()
        # span index table, to avoid redundant iterations
        span_2_id, id_2_span, ijss, ikcs, ikis, kjcs, kjis, basic_span = \
            constituent_index(sentence_length, False)
        # initial basic complete spans
        for ii in basic_span:
            (i, i, dir) = id_2_span[ii]
            complete_table[:, ii, :, :] = batch_decision_score[:, i, :, dir, :, 0]
            complete_table[:, ii, :, :] += batch_unary_score[:, i, :].unsqueeze(-1)
        for ij in ijss:
            (l, r, dir) = id_2_span[ij]
            num_ki = len(ikis[ij])
            ik_ci = complete_table[:, ikis[ij], :, :].reshape(batch_size, num_ki, tag_num, 1, valency_num)
            kj_ci = complete_table[:, kjis[ij], :, :].reshape(batch_size, num_ki, 1, tag_num, valency_num)
            # construct incomplete spans
            if dir == 0:
                span_i = ik_ci[:, :, :, :, 0].reshape(batch_size, num_ki, tag_num, 1, 1) \
                         + kj_ci[:, :, :, :, 1].reshape(batch_size, num_ki, 1, tag_num, 1) + \
                         right_score[:, r].transpose(1, 2).reshape(batch_size, 1, tag_num, tag_num,
                                                                                cvalency_num) \
                         + batch_decision_score[:, r, :, dir, :, 1].reshape(batch_size, 1, 1, tag_num, valency_num)
            else:
                span_i = ik_ci[:, :, :, :, 1].reshape(batch_size, num_ki, tag_num, 1, 1) \
                         + kj_ci[:, :, :, :, 0].reshape(batch_size, num_ki, 1, tag_num, 1) + \
                         left_score[:, l].reshape(batch_size, 1, tag_num, tag_num, cvalency_num) \
                         + batch_decision_score[:, l, :, dir, :, 1].reshape(batch_size, 1, tag_num, 1, valency_num)

            incomplete_table[:, ij, :, :, :], incomplete_backtrack[:, ij, :, :, :] \
                = torch.max(span_i, dim=1)
            # construct complete spans
            num_kc = len(ikcs[ij])
            if dir == 0:
                ik_cc = complete_table[:, ikcs[ij], :, :].reshape(batch_size, num_kc, tag_num, 1, valency_num)
                kj_ic = incomplete_table[:, kjcs[ij], :, :, :].reshape(batch_size, num_kc, tag_num, tag_num, valency_num)
                span_c = ik_cc[:, :, :, :, 0].reshape(batch_size, num_kc, tag_num, 1, 1) + kj_ic
                span_c = span_c.reshape(batch_size, num_kc * tag_num, tag_num, valency_num)
            else:
                ik_ic = incomplete_table[:, ikcs[ij], :, :, :].reshape(batch_size, num_kc, tag_num, tag_num, valency_num)
                kj_cc = complete_table[:, kjcs[ij], :, :].reshape(batch_size, num_kc, 1, tag_num, valency_num)
                span_c = ik_ic + kj_cc[:, :, :, :, 0].reshape(batch_size, num_kc, 1, tag_num, 1)
                span_c = span_c.transpose(2, 3).reshape(batch_size, num_kc * tag_num, tag_num, valency_num)
            complete_table[:, ij, :, :], complete_backtrack[:, ij, :, :] = \
                torch.max(span_c, dim=1)

        tags = tensor.new_zeros((batch_size, sentence_length)).int()
        heads = tensor.new_full((batch_size, sentence_length), -1).int()
        head_valences = tensor.new_zeros((batch_size, sentence_length)).int()
        valences = tensor.new_zeros((batch_size, sentence_length, 2)).int()
        for s, sent_len in enumerate(sent_lens.tolist()):
            root_id = span_2_id[(0, sent_len - 1, 1)]
            self._backtrack(incomplete_backtrack, complete_backtrack, root_id, 0, 0, 0, 1,
                            tags, heads, head_valences, valences,
                            ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, tag_num, s)

        heads = heads[:, 1:]
        return (heads, tags, head_valences, valences)

    def _backtrack(self, incomplete_backtrack, complete_backtrack, span_id, l_tag,
                   r_tag, decision_valence, complete, tags, heads, head_valences,
                   valences, ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, tag_num, sen_id):
        (l, r, dir) = id_2_span[span_id]
        if l == r:
            valences[sen_id, l, dir] = decision_valence
            return
        if complete:
            if dir == 0:
                k = complete_backtrack[sen_id, span_id, r_tag, decision_valence]
                # print 'k is ', k, ' complete left'
                k_span, k_tag = _divmod(k, tag_num)
                left_span_id = ikcs[span_id][k_span]
                right_span_id = kjcs[span_id][k_span]
                self._backtrack(incomplete_backtrack, complete_backtrack, left_span_id,
                                0, k_tag, 0, 1, tags, heads, head_valences, valences, ikcs,
                                ikis, kjcs, kjis, id_2_span, span_2_id, tag_num, sen_id)
                self._backtrack(incomplete_backtrack, complete_backtrack, right_span_id,
                                k_tag, r_tag, decision_valence, 0, tags, heads, head_valences,
                                valences, ikcs, ikis, kjcs, kjis, id_2_span, span_2_id,
                                tag_num, sen_id)
                return
            else:
                num_k = len(ikcs[span_id])
                k = complete_backtrack[sen_id, span_id, l_tag, decision_valence]
                # print 'k is ', k, ' complete right'
                k_span, k_tag = _divmod(k, tag_num)
                left_span_id = ikcs[span_id][k_span]
                right_span_id = kjcs[span_id][k_span]
                self._backtrack(incomplete_backtrack, complete_backtrack, left_span_id,
                                l_tag, k_tag, decision_valence, 0, tags, heads, head_valences,
                                valences, ikcs, ikis, kjcs, kjis, id_2_span, span_2_id,
                                tag_num, sen_id)
                self._backtrack(incomplete_backtrack, complete_backtrack, right_span_id,
                                k_tag, 0, 0, 1, tags, heads, head_valences, valences,
                                ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, tag_num, sen_id)
                return
        else:
            if dir == 0:

                k = incomplete_backtrack[sen_id, span_id, l_tag, r_tag, decision_valence]
                # print 'k is ', k, ' incomplete left'
                heads[sen_id, l] = r
                tags[sen_id, l] = l_tag
                head_valences[sen_id, l] = decision_valence
                left_span_id = ikis[span_id][k]
                right_span_id = kjis[span_id][k]
                self._backtrack(incomplete_backtrack, complete_backtrack, left_span_id,
                                l_tag, 0, 0, 1, tags, heads, head_valences, valences,
                                ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, tag_num, sen_id)
                self._backtrack(incomplete_backtrack, complete_backtrack, right_span_id,
                                0, r_tag, 1, 1, tags, heads, head_valences, valences,
                                ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, tag_num, sen_id)
                return
            else:
                k = incomplete_backtrack[sen_id, span_id, l_tag, r_tag, decision_valence]
                # print 'k is', k, ' incomplete right'
                heads[sen_id, r] = l
                tags[sen_id, r] = r_tag
                head_valences[sen_id, r] = decision_valence
                left_span_id = ikis[span_id][k]
                right_span_id = kjis[span_id][k]
                self._backtrack(incomplete_backtrack, complete_backtrack, left_span_id,
                                l_tag, 0, 1, 1, tags, heads, head_valences, valences,
                                ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, tag_num, sen_id)
                self._backtrack(incomplete_backtrack, complete_backtrack, right_span_id,
                                0, r_tag, 0, 1, tags, heads, head_valences, valences,
                                ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, tag_num, sen_id)
                return
