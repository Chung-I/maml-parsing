from typing import Dict, List, Any

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.models.ensemble import Ensemble
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.common import Params
from allennlp.data import Vocabulary

from src.models.biaffing_dependency_parser_vib import BiaffineDependencyParserMultiLangVIB

@Model.register("parser-ensemble")
class ParserEnsemble(Ensemble):
    """
    This class ensembles the output from multiple BiDAF models.

    It combines results from the submodels by averaging the start and end span probabilities.
    """

    def __init__(self, submodels: List[BiaffineDependencyParserMultiLangVIB]) -> None:
        super().__init__(submodels)

        self.model_class = BiaffineDependencyParserMultiLangVIB
        self._lang_attachment_scores: Dict[str, AttachmentScores] = defaultdict(AttachmentScores)

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
    ) -> Dict[str, torch.Tensor]:

        """
        The forward method runs each of the submodels, then selects the best from the subresults.
        """


        embedded_text_input = self.submodels[0]._embed(
            words, mask, pos_tags, metadata, lemmas, feats, langs
        )
        subresults = [
            submodel.get_arc_factored_probs(
                words, pos_tags, metadata, head_tags, head_indices, lemmas, feats,
                langs, return_metric, variational,
            )
            for submodel in self.submodels
        ]

        batch_size = subresults[0]["arc_log_probs"].size(0)

        predicted_heads, predicted_head_tags = ensemble(subresults)

        output_dict = {
            "hidden_states": None,
            "heads": predicted_heads,
            "head_tags": predicted_head_tags,
        }
        self.model_class._add_metadata_to_output_dict(metadata, output_dict)

        return output_dict

    # The logic here requires a custom from_params.
    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> "ParserEnsemble":  # type: ignore

        if vocab:
            raise ConfigurationError("vocab should be None")

        submodels = []
        paths = params.pop("submodels")
        for path in paths:
            submodels.append(load_archive(path).model)

        return cls(submodels=submodels)


def ensemble(subresults: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """
    Identifies the best prediction given the results from the submodels.

    Parameters
    ----------
    subresults : List[Dict[str, torch.Tensor]]
        Results of each submodel.

    Returns
    -------
    The index of the best submodel.
    """

    # Choose the highest average confidence span.

    stack_of_arc_logprobs = torch.stack([subresult["arc_log_probs"] for subresult in subresults])
    sum_arc_logprobs = torch.logsumexp(stack_of_arc_logprobs, dim=1)
    avg_arc_logprobs = sum_arc_logprobs + torch.log(torch.ones_like(sum_arc_logprobs) / len(subresults))

    stack_of_tag_logprobs = torch.stack([subresult["tag_log_probs"] for subresult in subresults])
    sum_tag_logprobs = torch.logsumexp(stack_of_tag_logprobs, dim=1)
    avg_tag_logprobs = sum_tag_logprobs + torch.log(torch.ones_like(sum_tag_logprobs) / len(subresults))
    batch_energy = torch.exp(avg_arc_logprobs.unsqueeze(1) + avg_tag_logprobs)

    return self.model_class._run_mst_decoding(batch_energy, subresults[0]["lengths"])
