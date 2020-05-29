import math
from typing import Optional, Tuple, Dict

from overrides import overrides
from transformers.configuration_auto import AutoConfig
from transformers.modeling_auto import AutoModel
from transformers.tokenization_auto import AutoTokenizer
import torch
import torch.nn.functional as F

from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules import InputVariationalDropout
from allennlp.nn.util import batched_index_select

from src.data.token_indexers import PretrainedAutoTokenizer
from src.modules.scalar_mix import ScalarMixWithDropout


class PretrainedAutoModel:
    """
    In some instances you may want to load the same BERT model twice
    (e.g. to use as a token embedder and also as a pooling layer).
    This factory provides a cache so that you don't actually have to load the model twice.
    """

    _cache: Dict[str, AutoModel] = {}

    @classmethod
    def load(cls, model_name: str, tokenizer_name: str, cache_model: bool = True,
             adapter_size: int = 8, pretrained: bool = True) -> AutoModel:
        has_adapter = False
        if model_name.startswith("adapter"):
            has_adapter = True
            _, model_name = model_name.split("_")

        if model_name in cls._cache:
            return PretrainedAutoModel._cache[model_name]

        pretrained_config = AutoConfig.from_pretrained(model_name,
                                                       output_hidden_states=True)

        if has_adapter:
            from src.modules.modeling_adapter_bert import AdapterBertModel
            pretrained_config.adapter_size = adapter_size
            model = AdapterBertModel.from_pretrained(model_name, config=pretrained_config)
        else:
            if pretrained:
                model = AutoModel.from_pretrained(model_name, config=pretrained_config)
            else:
                model = AutoModel.from_config(config=pretrained_config)

        if cache_model:
            cls._cache[model_name] = model

        return model


@TokenEmbedder.register("transformer")
class TransformerEmbedder(TokenEmbedder):
    """
    Uses a pretrained model from `transformers` as a `TokenEmbedder`.

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use. Should be the same as the corresponding
        `PretrainedTransformerIndexer`.
    max_length : `int`, optional (default = None)
        If positive, folds input token IDs into multiple segments of this length, pass them
        through the transformer model independently, and concatenate the final representations.
        Should be set to the same value as the `max_length` option on the
        `PretrainedTransformerIndexer`.
    """

    def __init__(self,
                 model_name: str,
                 max_length: int = None,
                 layer_dropout: float = 0.0,
                 bert_dropout: float = 0.0,
                 dropout: float = 0.0,
                 combine_layers: str = "mix",
                 adapter_size: int = 8,
                 pretrained: bool = True) -> None:
        super().__init__()
        placeholder = model_name.split("_")
        tokenizer_name = placeholder[-1]
        self.transformer_model = PretrainedAutoModel.load(model_name, tokenizer_name,
                                                          adapter_size=adapter_size,
                                                          pretrained=pretrained)
        self._max_length = max_length
        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.transformer_model.config.hidden_size
        self.combine_layers = combine_layers

        if self.combine_layers == "mix":
            self._scalar_mix = ScalarMixWithDropout(
                self.transformer_model.config.num_hidden_layers,
                do_layer_norm=False,
                dropout=layer_dropout
            )
        else:
            self._scalar_mix = None

        self._bert_dropout = InputVariationalDropout(bert_dropout)
        self.set_dropout(dropout)

        tokenizer = PretrainedAutoTokenizer.load(tokenizer_name)
        (
            self._num_added_start_tokens,
            self._num_added_end_tokens,
        ) = PretrainedTransformerIndexer.determine_num_special_tokens_added(tokenizer)
        self._num_added_tokens = self._num_added_start_tokens + self._num_added_end_tokens

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
        if (
            type_ids is not None
            and type_ids.max()
            >= self.transformer_model.embeddings.token_type_embeddings.num_embeddings
        ):
            raise ValueError("Found type ids too large for the chosen transformer model.")

        if type_ids is not None:
            assert token_ids.shape == type_ids.shape

        full_seq_len = token_ids.size(-1)
        too_long = full_seq_len > self._max_length

        if too_long:
            batch_size, num_segment_concat_wordpieces = token_ids.size()
            token_ids, segment_concat_mask, type_ids = self._fold_long_sequences(
                token_ids, segment_concat_mask, type_ids
            )

        transformer_mask = segment_concat_mask if too_long else mask
        # Shape: [batch_size, num_wordpieces, embedding_size],
        # or if self._max_length is not None:
        # [batch_size * num_segments, self._max_length, embedding_size]
        layer_outputs = self.transformer_model(
            input_ids=token_ids, token_type_ids=type_ids, attention_mask=transformer_mask
        )[-1][1:]
        layer_outputs = [self._bert_dropout(layer_output) for layer_output in layer_outputs]
        if self._scalar_mix is not None:
            embeddings = self._scalar_mix(layer_outputs, transformer_mask)
        elif self.combine_layers == "last":
            embeddings = layer_outputs[-1]
        else:
            raise NotImplementedError

        if too_long:
            embeddings = self._unfold_long_sequences(
                embeddings, segment_concat_mask, batch_size, num_segment_concat_wordpieces
            )

        return embeddings

    def set_dropout(self, dropout):
        """
        Applies dropout to all transformer layers
        """
        self.dropout = dropout

        self.transformer_model.embeddings.dropout.p = dropout

        for layer in self.transformer_model.encoder.layer:
            layer.attention.self.dropout.p = dropout
            layer.attention.output.dropout.p = dropout
            layer.output.dropout.p = dropout

    def _fold_long_sequences(
        self,
        token_ids: torch.LongTensor,
        mask: torch.LongTensor,
        type_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.LongTensor, torch.LongTensor, Optional[torch.LongTensor]]:
        """
        We fold 1D sequences (for each element in batch), returned by `PretrainedTransformerIndexer`
        that are in reality multiple segments concatenated together, to 2D tensors, e.g.

        [ [CLS] A B C [SEP] [CLS] D E [SEP] ]
        -> [ [ [CLS] A B C [SEP] ], [ [CLS] D E [SEP] [PAD] ] ]
        The [PAD] positions can be found in the returned `mask`.

        # Parameters

        token_ids: `torch.LongTensor`
            Shape: [batch_size, num_segment_concat_wordpieces].
            num_segment_concat_wordpieces is num_wordpieces plus special tokens inserted in the
            middle, i.e. the length of: "[CLS] A B C [SEP] [CLS] D E F [SEP]" (see indexer logic).
        mask: `torch.LongTensor`
            Shape: [batch_size, num_segment_concat_wordpieces].
            The mask for the concatenated segments of wordpieces. The same as `segment_concat_mask`
            in `forward()`.
        type_ids: Optional[torch.LongTensor]
            Shape: [batch_size, num_segment_concat_wordpieces].

        # Returns:

        token_ids: `torch.LongTensor`
            Shape: [batch_size * num_segments, self._max_length].
        mask: `torch.LongTensor`
            Shape: [batch_size * num_segments, self._max_length].
        """
        num_segment_concat_wordpieces = token_ids.size(1)
        num_segments = math.ceil(num_segment_concat_wordpieces / self._max_length)
        padded_length = num_segments * self._max_length
        length_to_pad = padded_length - num_segment_concat_wordpieces

        def fold(tensor):  # Shape: [batch_size, num_segment_concat_wordpieces]
            # Shape: [batch_size, num_segments * self._max_length]
            tensor = F.pad(tensor, [0, length_to_pad], value=0)
            # Shape: [batch_size * num_segments, self._max_length]
            return tensor.reshape(-1, self._max_length)

        return fold(token_ids), fold(mask), fold(type_ids) if type_ids is not None else None

    def _unfold_long_sequences(
        self,
        embeddings: torch.FloatTensor,
        mask: torch.LongTensor,
        batch_size: int,
        num_segment_concat_wordpieces: int,
    ) -> torch.FloatTensor:
        """
        We take 2D segments of a long sequence and flatten them out to get the whole sequence
        representation while remove unnecessary special tokens.

        [ [ [CLS]_emb A_emb B_emb C_emb [SEP]_emb ], [ [CLS]_emb D_emb E_emb [SEP]_emb [PAD]_emb ] ]
        -> [ [CLS]_emb A_emb B_emb C_emb D_emb E_emb [SEP]_emb ]

        We truncate the start and end tokens for all segments, recombine the segments,
        and manually add back the start and end tokens.

        # Parameters

        embeddings: `torch.FloatTensor`
            Shape: [batch_size * num_segments, self._max_length, embedding_size].
        mask: `torch.LongTensor`
            Shape: [batch_size * num_segments, self._max_length].
            The mask for the concatenated segments of wordpieces. The same as `segment_concat_mask`
            in `forward()`.
        batch_size: `int`
        num_segment_concat_wordpieces: `int`
            The length of the original "[ [CLS] A B C [SEP] [CLS] D E F [SEP] ]", i.e.
            the original `token_ids.size(1)`.

        # Returns:

        embeddings: `torch.FloatTensor`
            Shape: [batch_size, self._num_wordpieces, embedding_size].
        """

        def lengths_to_mask(lengths, max_len, device):
            return torch.arange(max_len, device=device).expand(
                lengths.size(0), max_len
            ) < lengths.unsqueeze(1)

        device = embeddings.device
        num_segments = int(embeddings.size(0) / batch_size)
        embedding_size = embeddings.size(2)

        # We want to remove all segment-level special tokens but maintain sequence-level ones
        num_wordpieces = num_segment_concat_wordpieces - (num_segments - 1) * self._num_added_tokens

        embeddings = embeddings.reshape(batch_size, num_segments * self._max_length, embedding_size)
        mask = mask.reshape(batch_size, num_segments * self._max_length)
        # We assume that all 1s in the mask preceed all 0s, and add an assert for that.
        # Open an issue on GitHub if this breaks for you.
        # Shape: (batch_size,)
        seq_lengths = mask.sum(-1)
        if not (lengths_to_mask(seq_lengths, mask.size(1), device) == mask).all():
            raise ValueError(
                "Long sequence splitting only supports masks with all 1s preceding all 0s."
            )
        # Shape: (batch_size, self._num_added_end_tokens); this is a broadcast op
        end_token_indices = (
            seq_lengths.unsqueeze(-1) - torch.arange(self._num_added_end_tokens, device=device) - 1
        )

        # Shape: (batch_size, self._num_added_start_tokens, embedding_size)
        start_token_embeddings = embeddings[:, : self._num_added_start_tokens, :]
        # Shape: (batch_size, self._num_added_end_tokens, embedding_size)
        end_token_embeddings = batched_index_select(embeddings, end_token_indices)

        embeddings = embeddings.reshape(batch_size, num_segments, self._max_length, embedding_size)
        embeddings = embeddings[
            :, :, self._num_added_start_tokens : -self._num_added_end_tokens, :
        ]  # truncate segment-level start/end tokens
        embeddings = embeddings.reshape(batch_size, -1, embedding_size)  # flatten

        # Now try to put end token embeddings back which is a little tricky.

        # The number of segment each sequence spans, excluding padding. Mimicking ceiling operation.
        # Shape: (batch_size,)
        num_effective_segments = (seq_lengths + self._max_length - 1) / self._max_length
        # The number of indices that end tokens should shift back.
        num_removed_non_end_tokens = (
            num_effective_segments * self._num_added_tokens - self._num_added_end_tokens
        )
        # Shape: (batch_size, self._num_added_end_tokens)
        end_token_indices -= num_removed_non_end_tokens.unsqueeze(-1)
        assert (end_token_indices >= self._num_added_start_tokens).all()
        # Add space for end embeddings
        embeddings = torch.cat([embeddings, torch.zeros_like(end_token_embeddings)], 1)
        # Add end token embeddings back
        embeddings.scatter_(
            1, end_token_indices.unsqueeze(-1).expand_as(end_token_embeddings), end_token_embeddings
        )

        # Now put back start tokens. We can do this before putting back end tokens, but then
        # we need to change `num_removed_non_end_tokens` a little.
        embeddings = torch.cat([start_token_embeddings, embeddings], 1)

        # Truncate to original length
        embeddings = embeddings[:, :num_wordpieces, :]
        return embeddings

@TokenEmbedder.register("transformer_pretrained")
class PretrainedTransformerEmbedder(TransformerEmbedder):

    """
    # Parameters

    pretrained_model : `str`
        Either the name of the pretrained model to use (e.g. 'bert-base-uncased'),
        or the path to the .tar.gz file with the model weights.

        If the name is a key in the list of pretrained models at
        https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bert.py#L34
        the corresponding path will be used; otherwise it will be interpreted as a path or URL.
    requires_grad : `bool`, optional (default = False)
        If True, compute gradient of BERT parameters for fine tuning.
    top_layer_only : `bool`, optional (default = `False`)
        If `True`, then only return the top layer instead of apply the scalar mix.
    scalar_mix_parameters : `List[float]`, optional, (default = None)
        If not `None`, use these scalar mix parameters to weight the representations
        produced by different layers. These mixing weights are not updated during
        training.
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
    ) -> None:

        super().__init__(
            model_name=model_name,
            max_length=max_length,
            layer_dropout=layer_dropout,
            bert_dropout=bert_dropout,
            dropout=dropout,
            combine_layers=combine_layers,
            adapter_size=adapter_size,
            pretrained=pretrained,
        )
        for name, param in self.transformer_model.named_parameters():
            if model_name.startswith("adapter") and 'adapter' in name:
                param.requires_grad = True
            else:
                param.requires_grad = requires_grad
