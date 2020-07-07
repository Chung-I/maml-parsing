from typing import Optional

from overrides import overrides
import torch

from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.nn import util

from src.modules.token_embedders import PretrainedTransformerEmbedder


@TokenEmbedder.register("transformer_pretrained_mismatched")
class PretrainedTransformerMismatchedEmbedder(TokenEmbedder):
    """
    Use this embedder to embed wordpieces given by `PretrainedTransformerMismatchedIndexer`
    and to pool the resulting vectors to get word-level representations.

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use. Should be the same as the corresponding
        `PretrainedTransformerMismatchedIndexer`.
    max_length : `int`, optional (default = None)
        If positive, folds input token IDs into multiple segments of this length, pass them
        through the transformer model independently, and concatenate the final representations.
        Should be set to the same value as the `max_length` option on the
        `PretrainedTransformerMismatchedIndexer`.
    """

    def __init__(
        self,
        model_name: str,
        max_length: int = None,
        requires_grad: bool = True,
        layer_dropout: float = 0.0,
        bert_dropout: float = 0.0,
        dropout: float = 0.0,
        combine_layers: str = "mix",
        adapter_size: int = 8,
        pretrained: bool = True,
        mean_affix: str = None,
    ) -> None:
        super().__init__()
        # The matched version v.s. mismatched
        self._matched_embedder = PretrainedTransformerEmbedder(
            model_name,
            max_length=max_length,
            requires_grad=requires_grad,
            layer_dropout=layer_dropout,
            bert_dropout=bert_dropout,
            dropout=dropout,
            combine_layers=combine_layers,
            adapter_size=adapter_size,
            pretrained=pretrained,
            mean_affix=mean_affix,
        )

    @overrides
    def get_output_dim(self):
        return self._matched_embedder.get_output_dim()

    @overrides
    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.LongTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.LongTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.LongTensor] = None,
        lang: Optional[str] = None,
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters

        token_ids: torch.LongTensor
            Shape: [batch_size, num_wordpieces] (for exception see `PretrainedTransformerEmbedder`).
        mask: torch.LongTensor
            Shape: [batch_size, num_orig_tokens].
        offsets: torch.LongTensor
            Shape: [batch_size, num_orig_tokens, 2].
            Maps indices for the original tokens, i.e. those given as input to the indexer,
            to a span in token_ids. `token_ids[i][offsets[i][j][0]:offsets[i][j][1] + 1]`
            corresponds to the original j-th token from the i-th batch.
        wordpiece_mask: torch.LongTensor
            Shape: [batch_size, num_wordpieces].
        type_ids: Optional[torch.LongTensor]
            Shape: [batch_size, num_wordpieces].
        segment_concat_mask: Optional[torch.LongTensor]
            See `PretrainedTransformerEmbedder`.

        # Returns:

        Shape: [batch_size, num_orig_tokens, embedding_size].
        """
        # Shape: [batch_size, num_wordpieces, embedding_size].
        embeddings = self._matched_embedder(
            token_ids, wordpiece_mask, type_ids=type_ids, segment_concat_mask=segment_concat_mask, lang=lang,
        )

        # span_embeddings: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        # span_mask: (batch_size, num_orig_tokens, max_span_length)
        span_embeddings, span_mask = util.batched_span_select(embeddings.contiguous(), offsets)
        span_mask = span_mask.unsqueeze(-1)
        span_embeddings *= span_mask  # zero out paddings

        span_embeddings_sum = span_embeddings.sum(2)
        span_embeddings_len = span_mask.sum(2)
        # Shape: (batch_size, num_orig_tokens, embedding_size)
        orig_embeddings = span_embeddings_sum / span_embeddings_len

        return orig_embeddings
