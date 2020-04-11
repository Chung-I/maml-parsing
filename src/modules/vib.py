'''
    Xiang Li
    xli150@jhu.edu
'''
from overrides import overrides

import numpy as np
from torch import nn
import torch
import torch.optim as optim

from allennlp.common import FromParams
from allennlp.nn.activations import Activation

SMALL = 1e-10
class ContinuousEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        tag_dim: int,
        activation: str,
        embedding_dim: int,
    ):
        super(ContinuousEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.tag_dim = tag_dim
        self.activation = Activation.by_name(activation)()
        self.embedding_dim = embedding_dim

        # ============= Covariance matrix & Mean vector ================
        interm_layer_size = (self.embedding_dim + self.hidden_dim) // 2
        self.linear_layer = nn.Linear(self.embedding_dim, interm_layer_size )
        self.linear_layer3 = nn.Linear(interm_layer_size, self.hidden_dim)

        self.hidden2mean = nn.Linear(self.hidden_dim, self.tag_dim)
        self.hidden2std = nn.Linear(self.hidden_dim, self.tag_dim)

    def forward_sent(self, sent, elmo_embeds, index=None):

        ''' used for some evaluation scripts, not for training '''
        sent_len = len(sent)
        embeds = elmo_embeds[index]
        temps = self.activation(self.linear_layer(embeds))
        temps = self.activation(self.linear_layer3(temps))
        mean = self.hidden2mean(temps)
        std = self.hidden2std(temps)
        std = std.view(sent_len, 1, self.tag_dim)
        cov_lst = []
        cov_line = []
        cov_line = std.view(-1)
        cov_line = cov_line * cov_line + SMALL
        return mean, cov_lst, cov_line


    def forward_sent_batch(self, embeds):

        temps = self.activation(self.linear_layer(embeds))
        temps = self.activation(self.linear_layer3(temps))
        mean = self.hidden2mean(temps) # bsz, seqlen, dim
        std = self.hidden2std(temps) # bsz, seqlen, dim
        cov = std * std + SMALL
        return mean, cov

    def get_sample_from_param_batch(self, mean, cov, sample_size):
        bsz, seqlen, tag_dim =  mean.shape
        z = torch.randn(bsz, sample_size, seqlen, tag_dim, device=mean.device)
        z = z * torch.sqrt(cov).unsqueeze(1).expand(-1, sample_size, -1, -1) + \
            mean.unsqueeze(1).expand(-1, sample_size, -1, -1)
        return z.view(-1, seqlen, tag_dim)


    def get_statistics_batch(self, elmo_embeds):
        mean, cov = self.forward_sent_batch(elmo_embeds)
        return mean, cov


class ContinuousVIB(nn.Module, FromParams):
    '''
        this is the primary class for this bottleneck model.
        enjoy and have fun !
    '''

    def __init__(
        self,
        tag_dim: int,
        embedding_dim: int,
        encoder_output_dim: int,
        activation: str = "elu",
        max_sent_len: int = 512,
        beta: float = 0.1,
        sample_size: int = 5,
        sample_method: str = "iid",
        type_token_reg: bool = True,
        gamma: float = -1,
        ):

        super(ContinuousVIB, self).__init__()

        # ===============Param setup===================
        self.beta = beta
        self.max_sent_len = max_sent_len
        self.tag_dim = tag_dim
        self.embedding_dim = embedding_dim
        self.encoder_output_dim = encoder_output_dim
        self.hidden_dim = encoder_output_dim
        self.sample_size = sample_size
        self.sample_method = sample_method
        self.type_token_reg = type_token_reg

        # Annealing parameters. currently set to FALSE
        self.anneal_rate = 0.0005
        self.temperature = 5
        self.min_temp = 0.5
        self.min_inv_gamma = 0.1
        self.min_inv_beta = 0.1
        self.beta_annealing = False
        self.gamma_annealing = False

        self.gamma = gamma if self.type_token_reg else 0.0

        # =============== Encoder Decoder setup ==================
        self.encoder = ContinuousEncoder(
            self.hidden_dim,
            self.tag_dim,
            activation,
            self.encoder_output_dim,
        )
        ## TODO: could swap the decoder here.
        if self.type_token_reg:
            self.variational_encoder = ContinuousEncoder(
                self.hidden_dim,
                self.tag_dim,
                activation,
                self.embedding_dim,
            )


    def kl_div(self, param1, param2, mask):
        """
        Calculates the KL divergence between a categorical distribution and a
        uniform categorical distribution.
        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the categorical or gumbel-softmax distribution.
            Shape (N, D)
        """
        mean1, cov1 = param1
        mean2, cov2 = param2
        ones = torch.ones_like(mean1)

        bsz, seqlen, tag_dim = mean1.shape
        var_len = tag_dim * seqlen

        cov2_inv = 1 / cov2
        mean_diff = mean1 - mean2

        KL = -0.5 * (torch.log(cov1) - torch.log(cov2) - cov1 * cov2_inv
                     + ones - mean_diff * mean_diff * cov2_inv)
        KL = KL.sum(dim=-1)
        KL = KL * mask.float()
        KL = KL.sum() / mask.float().sum()
        return KL

    @overrides
    def forward(self, head_indices, head_tags, pos_tags, mask, r_mean, r_std, sample_size=None,
                sample_method=None, type_embeds=None, non_context_embeds=None):
        if sample_method is None:
            sample_method = self.sample_method
        if sample_size is None:
            sample_size = self.sample_size
        total_loss = 0
        mean, cov = self.encoder.get_statistics_batch(type_embeds)
        bsz, seqlen, _ = type_embeds.shape

        original_mask = mask
        if sample_method == "argmax":
            t = mean
        elif sample_method == 'iid':
            t = self.encoder.get_sample_from_param_batch(mean, cov, sample_size)
            head_indices = head_indices.unsqueeze(1).repeat(1, sample_size, 1).view(bsz * sample_size, seqlen)
            head_tags = head_tags.unsqueeze(1).repeat(1, sample_size, 1).view(bsz * sample_size, seqlen)
            mask = mask.unsqueeze(1).repeat(1, sample_size, 1).view(bsz * sample_size, seqlen)
            pos_tags = pos_tags.unsqueeze(1).repeat(1, sample_size, 1).view(bsz * sample_size, seqlen)
        else:
            print('missing option for sample_tag, double check')


        if seqlen <= self.max_sent_len:
            mean_r = r_mean[:, :seqlen]
            std_r = r_std[:, :seqlen]
        else:
            div, rem = divmod(self.max_sent_len, seqlen)
            mean_r = torch.cat([r_mean] * div + [r_mean[:,:rem]], dim=1)
            std_r = torch.cat([r_std] * div + [r_std[:,:rem]], dim=1)
        cov_r = std_r * std_r + SMALL
        kl_div = self.kl_div((mean, cov), (mean_r, cov_r), original_mask)
        total_loss += self.beta * kl_div

        if self.type_token_reg:
            mean2, cov2 = self.variational_encoder.get_statistics_batch(non_context_embeds)
            kl_div2 = self.kl_div((mean, cov), (mean2, cov2), original_mask)
            total_loss += self.gamma * kl_div2
        else:
            kl_div2 = torch.tensor([-1.])

        return t, head_indices, head_tags, pos_tags, mask, total_loss, \
            kl_div.item(), kl_div2.item()


    def anneal_clustering(self, decrease_rate, tag='beta'):
        '''
        This function aims to do annealing and gradually do more compression, by tuning the gamma and beta
        to be larger. So, this is equivalent as annealing the inverse of the beta and gamma, and make them
        smaller, we decide that the lower limit of this annealing is when beta = 10, that is, inv_beta = 0.1

        :param decrease_rate:
        :param tag:
        :return:
        '''
        if tag == 'beta':
            inv_beta = 1/self.beta
            inv_beta = np.maximum(inv_beta * np.exp(-decrease_rate), self.min_inv_beta)
            self.beta = np.asscalar(1/inv_beta)
        elif tag == 'gamma':
            inv_gamma = 1 / self.gamma
            inv_gamma = np.maximum(inv_gamma * np.exp(-decrease_rate), self.min_inv_gamma)
            self.gamma = np.asscalar(1 / inv_gamma)


    def train_batch(self, corpus, sent_per_epoch, elmo_embeds,
                        non_context_embeds, delta_temp=0.01 , tag=''):

        shuffledData = corpus
        shuffle_indices = np.random.choice(len(shuffledData), min(sent_per_epoch, len(shuffledData)), replace=False)
        epoch_loss = 0
        batch_total = 0

        align_err_total, nlogp_total, word_total, sent_total, kl_total, kl_total2, label_LAS_total = 0,0,0,0,0,0,0
        for iSentence, ind in enumerate(shuffle_indices):
            x, tag_, y, y_label = shuffledData[ind]

            bsz, seqlen = x.shape

            result, err_total, accuracy_loss, length_total, sample_total, kl_loss, kl_loss2, label_LAS = \
                self.forward_batch((x, y, y_label), type_embeds=elmo_embeds_, non_context_embeds=non_context_embeds_)
            # average per batch, actually per token.
            align_err_total += err_total
            batch_total += 1
            nlogp_total += accuracy_loss
            label_LAS_total += label_LAS
            kl_total += kl_loss

            # average per sentence.
            word_total += length_total * bsz
            sent_total += bsz
            kl_total2 += kl_loss2

            result.backward()
            self.optimizer_decoder.step()
            self.optimizer_encoder.step()
            if self.type_token_reg:
                self.optimizer_var.step()
                self.optimizer_var.zero_grad()
            self.optimizer_decoder.zero_grad()
            self.optimizer_encoder.zero_grad()
            epoch_loss += result.item()

            ''' min(args.kl_pen + args.delta_kl, 1) '''
            self.temperature = np.maximum(self.temperature - delta_temp, self.min_temp)
            self.encoder.temperature = self.temperature

        avg_seqlen = word_total / sent_total
        align_err_w = align_err_total / batch_total
        nlogp_w = nlogp_total / batch_total
        align_err_s = align_err_w * avg_seqlen
        nlogp_s = nlogp_w * avg_seqlen
        kl_s = kl_total / batch_total
        kl_s2 = kl_total2 / batch_total
        kl_w = kl_s / avg_seqlen
        kl_w2 = kl_s2 / avg_seqlen
        LAS = label_LAS_total / batch_total

        print(
            'Total: totalLoss_per_sent=%f, NLL=%.3f, KL=%.3f, KL2=%.3f, UAS=%.3f LAS=%.3f, beta=%f, gamma=%f, temp=%f'
            % (epoch_loss / sent_total, nlogp_s, kl_s, kl_s2, 1 - align_err_w, LAS, self.beta, self.gamma,
               self.temperature))

        result_dict = {}
        result_dict["align_err_w"] = align_err_w
        result_dict["nlogp_w"] = nlogp_w
        result_dict["align_err_s"] = align_err_s
        result_dict["nlogp_s"] = nlogp_s
        result_dict["kl_s"] = kl_s
        result_dict["kl_s2"] = kl_s2
        result_dict["kl_w"] = kl_w
        result_dict["kl_w2"] = kl_w2

        result_dict["LAS"] = LAS

        return result_dict
