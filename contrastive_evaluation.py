import argparse
import pathlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

from translation_models import ScoringModel


@dataclass
class MTContrastiveEvaluationResult:
    accuracy: float
    discrepancy: Optional[float]

    def __str__(self):
        s = f"Accuracy: {100 * self.accuracy:.1f}"
        if self.discrepancy is not None:
            s += f"\nDistributional discrepancy: {self.discrepancy:.1f}"
        return s


class MTContrastiveEvaluationTask:

    def __init__(self,
                 src_path: Path,
                 ref_path: Path,
                 contrastive_path: Path,
                 ):
        with open(src_path) as f:
            self.sources = f.read().splitlines()
        with open(ref_path) as f:
            self.references = f.read().splitlines()
        with open(contrastive_path) as f:
            self.contrastive_translations = f.read().splitlines()
        assert len(self.sources) == len(self.references) == len(self.contrastive_translations)

    @torch.no_grad()
    def evaluate(self,
                 translation_model: ScoringModel = None,
                 scores: List[float] = None,
                 compute_testset_metrics: bool = False,
                 ):
        """
        Provide either `translation_model` or a list of `scores`.
        :param translation_model: An instance of `ScoringModel` that can score hypotheses given a source
        :param scores: A list of scores for the test set (first the scores for the correct hypotheses, then the scores for the incorrect hypotheses)
        :param compute_testset_metrics: Report the discrepancy of the test set w.r.t the `translation_model` (takes some time).
        """
        if (translation_model is not None and scores is not None) or (translation_model is None and scores is None):
            raise ValueError("You should provide either a translation model or a list of scores")
        if compute_testset_metrics and translation_model is None:
            raise ValueError("A translation model is required to compute test set metrics")

        if scores is not None:
            assert len(scores) == 2 * len(self.sources), "Number of scores needs to match test set size (first the scores for the correct hypotheses, then the scores for the incorrect hypotheses)"
            reference_scores = scores[:len(self.sources)]
            contrastive_scores = scores[len(self.sources):]
        else:
            reference_scores = translation_model.score(self.sources, self.references)
            contrastive_scores = translation_model.score(self.sources, self.contrastive_translations)

        num_correct = sum(reference_score >= contrastive_score for reference_score, contrastive_score in zip(reference_scores, contrastive_scores))
        accuracy = num_correct / len(self.sources)

        if compute_testset_metrics:
            translations = translation_model.translate(self.sources)
            actual_scores = translation_model.score(self.sources, translations)
            reference_scores = torch.Tensor(reference_scores)
            contrastive_scores = torch.Tensor(contrastive_scores)
            actual_scores = torch.Tensor(actual_scores)
            better_scores = torch.maximum(reference_scores, contrastive_scores)
            # In rare cases, the actual score is lower than the "better score"
            distance = torch.maximum(torch.zeros_like(actual_scores), actual_scores - better_scores)
            discrepancy = distance[~torch.isnan(distance)].mean().item()
        else:
            discrepancy = None

        result = MTContrastiveEvaluationResult(
            accuracy=accuracy,
            discrepancy=discrepancy,
        )
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset-name')
    parser.add_argument('--scores-path', type=pathlib.Path)
    args = parser.parse_args()

    testset_path = Path(__file__).parent / "data" / args.testset_name
    assert testset_path.exists()
    task = MTContrastiveEvaluationTask(
        src_path=testset_path / "src.en",
        ref_path=testset_path / "tgt.correct.de",
        contrastive_path=testset_path / "tgt.incorrect.de",
    )
    with open(args.scores_path) as f:
        scores = list(map(float, f))
    result = task.evaluate(
        scores=scores,
    )
    print(result)
