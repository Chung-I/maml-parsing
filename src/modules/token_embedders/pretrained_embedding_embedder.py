import math
from typing import Optional, Tuple

from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules import Seq2SeqEncoder


@TokenEmbedder.register("pretrained_embedding")
class PretrainedEmbeddingEmbedder(TokenEmbedder):
    """
    Uses a pretrained model from `transformers` as a `TokenEmbedder`.

    # Parameters

    """

    def __init__(
        self,
        pretrained_file: str,
        encoder: Seq2SeqEncoder = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        weight = torch.load(pretrained_file)
        self.embeddings = nn.Embedding.from_pretrained(weight, freeze=(not trainable))
        self.encoder = encoder
        self.output_dim = self.embeddings.embedding_dim

    @overrides
    def get_output_dim(self):
        return self.output_dim

    @overrides
    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.LongTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters

        token_ids: torch.LongTensor
            Shape: [
                batch_size, num_wordpieces if max_length is None else num_segment_concat_wordpieces
            ].
            num_segment_concat_wordpieces is num_wordpieces plus special tokens inserted in the
            middle, e.g. the length of: "[CLS] A B C [SEP] [CLS] D E F [SEP]" (see indexer logic).
        mask: torch.LongTensor
            Shape: [batch_size, num_wordpieces].
        type_ids: Optional[torch.LongTensor]
            Shape: [
                batch_size, num_wordpieces if max_length is None else num_segment_concat_wordpieces
            ].
        segment_concat_mask: Optional[torch.LongTensor]
            Shape: [batch_size, num_segment_concat_wordpieces].

        # Returns:

        Shape: [batch_size, num_wordpieces, embedding_size].
        """
        embedding = self.embeddings(token_ids)
        if self.encoder is not None:
            encoded_text = self.encoder(embedding)
            return encoded_text

        return embedding

