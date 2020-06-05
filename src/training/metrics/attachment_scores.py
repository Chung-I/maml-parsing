from typing import Optional, List

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric

@Metric.register("uas")
class AttachmentScores(Metric):
    """
    Computes labeled and unlabeled attachment scores for a
    dependency parse, as well as sentence level exact match
    for both labeled and unlabeled trees. Note that the input
    to this metric is the sampled predictions, not the distribution
    itself.

    Parameters
    ----------
    ignore_classes : ``List[int]``, optional (default = None)
        A list of label ids to ignore when computing metrics.
    """

    def __init__(self) -> None:
        self._unlabeled_correct = 0.0
        self._exact_unlabeled_correct = 0.0
        self._total_words = 0.0
        self._total_sentences = 0.0

    def __call__(  # type: ignore
        self,
        predicted_indices: torch.Tensor,
        gold_indices: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        predicted_indices : ``torch.Tensor``, required.
            A tensor of head index predictions of shape (batch_size, timesteps).
        predicted_labels : ``torch.Tensor``, required.
            A tensor of arc label predictions of shape (batch_size, timesteps).
        gold_indices : ``torch.Tensor``, required.
            A tensor of the same shape as ``predicted_indices``.
        gold_labels : ``torch.Tensor``, required.
            A tensor of the same shape as ``predicted_labels``.
        mask: ``torch.Tensor``, optional (default = None).
            A tensor of the same shape as ``predicted_indices``.
        """
        unwrapped = self.unwrap_to_tensors(
            predicted_indices, gold_indices, mask
        )
        predicted_indices, gold_indices, mask = unwrapped

        mask = mask.long()
        predicted_indices = predicted_indices.long()
        gold_indices = gold_indices.long()

        correct_indices = predicted_indices.eq(gold_indices).long() * mask
        unlabeled_exact_match = (correct_indices + (1 - mask)).prod(dim=-1)

        self._unlabeled_correct += correct_indices.sum()
        self._exact_unlabeled_correct += unlabeled_exact_match.sum()
        self._total_sentences += correct_indices.size(0)
        self._total_words += correct_indices.numel() - (1 - mask).sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated metrics as a dictionary.
        """
        unlabeled_attachment_score = 0.0
        unlabeled_exact_match = 0.0
        if self._total_words > 0.0:
            unlabeled_attachment_score = float(
                self._unlabeled_correct) / float(self._total_words)
        if self._total_sentences > 0:
            unlabeled_exact_match = float(self._exact_unlabeled_correct) / float(
                self._total_sentences
            )
        if reset:
            self.reset()
        return {
            "UAS": unlabeled_attachment_score,
            "UEM": unlabeled_exact_match,
        }

    def reset(self):
        self._unlabeled_correct = 0.0
        self._exact_unlabeled_correct = 0.0
        self._total_words = 0.0
        self._total_sentences = 0.0
