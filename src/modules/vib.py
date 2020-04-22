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

from src.training.util import move_to_device, flatten

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
        self.hidden_dim = int((embedding_dim + tag_dim) / 2)
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
            self.embedding_dim,
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

    @staticmethod
    def get_kl_loss(generator, tasks):
        def get_hidden_states(task):
            kl_losses = []
            kl_divs = []
            kl_div2s = []

            device = next(generator.parameters()).device
            for inputs in task:
                inputs = move_to_device(inputs, device)
                output_dict = generator(**inputs, variational=True)
                kl_loss = output_dict["kl_loss"]
                kl_div = output_dict["kl_div"]
                kl_div2 = output_dict["kl_div2"]
                kl_losses.append(kl_loss)
                kl_divs.append(kl_div)
                kl_div2s.append(kl_div2)

            return kl_losses, kl_divs, kl_div2s

        kl_losses, kl_divs, kl_div2s = map(flatten, zip(*map(get_hidden_states, tasks)))

        kl_loss = sum(kl_losses) / len(kl_losses)
        kl_div = sum(kl_divs) / len(kl_divs)
        kl_div2 = sum(kl_div2s) / len(kl_div2s)
        return kl_loss, kl_div, kl_div2
