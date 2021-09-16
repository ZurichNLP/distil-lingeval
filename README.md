
# DistilLingEval

Data and code for the paper ["On the Limits of Minimal Pairs in Contrastive Evaluation"](https://arxiv.org/abs/2109.07465) (BlackboxNLP 2021), containing contrastive translation pairs for targeted evaluation of **English→German** MT systems.

The evaluation protocol is identical to LingEval97 (https://github.com/rsennrich/lingeval97).
The difference is that the target sequences of LingEval97 are human-written references, whereas DistilLingEval also provides **contrastive test sets built from machine translations**.

## Contrastive Test Sets

The table below compares the phenomena or error types that are covered by the different test set variants:

<table>
<thead>
  <tr>
    <th rowspan="2">Phenomenon</th>
    <th rowspan="2"><a href="https://github.com/rsennrich/lingeval97">LingEval97</a></th>
    <th colspan="2">DistilLingEval (this repo)</th>
  </tr>
  <tr>
    <th>Human references</th>
    <th>Machine references</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>auxiliary</td>
    <td>✓</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>compound</td>
    <td>✓</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>np_agreement</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
  </tr>
  <tr>
    <td>polarity_affix_del</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
  </tr>
  <tr>
    <td>polarity_affix_ins</td>
    <td>✓</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>polarity_particle_kein_del</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
  </tr>
  <tr>
    <td>polarity_particle_kein_ins</td>
    <td>✓</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>polarity_particle_nicht_del</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
  </tr>
  <tr>
    <td>polarity_particle_nicht_ins</td>
    <td>✓</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>subj_adequacy</td>
    <td>✓</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>subj_verb_agreement</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
  </tr>
  <tr>
    <td>transliteration</td>
    <td>✓</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>verb_particle</td>
    <td>✓</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>clause_omission</td>
    <td></td>
    <td>✓</td>
    <td>✓</td>
  </tr>
  <tr>
    <td>hypercorrect_genitive<sup>1, 2</sup></td>
    <td></td>
    <td>✓</td>
    <td>✓</td>
  </tr>
  <tr>
    <td>placeholder_ding<sup>1</sup></td>
    <td></td>
    <td>✓</td>
    <td>✓</td>
  </tr>
</tbody>
</table>

Some remarks:
1. The `hypercorrect_genitive` and `placeholder_ding` test sets are based on implausible hypotheses – they were created to demonstrate how human references and machine references can lead to different evaluation results. Beyond that, the two test sets are not too useful, since everything below very high accuracy would be a surprise.
2. The `hypercorrect_genitive` test sets are based on a variety of parallel corpora.
3. Otherwise, LingEval97 and DistilLingEval draw from the same distribution ([wmt09–wmt16](https://github.com/mjpost/sacrebleu/blob/master/DATASETS.md) test sets). However, LingEval97 and the human-reference variants of DistilLingEval do not overlap perfectly because different implementations have been used to select sentence pairs and to create contrastive variants.

### Which test set variant should I use?

- Use **DistilLingEval with machine references** if you are interested in the likely behavior of your system.
- Use **DistilLingEval with human references** to evaluate the robustness of your system against hypotheses or target contexts written by humans.
- Use **LingEval97** to compare your system to previous work that reports LingEval97 results, or to analyze linguistic phenomena that are only covered by LingEval97.

## Running the Evaluation

### Installation

- Requires Python >= 3.7
- Requires PyTorch (tested with 1.9.0)
- `pip install -r requirements.txt`
- Optional dependencies for Fairseq models:
    - fairseq==0.10.2
    - fastBPE==0.1.0
    - sacremoses==0.0.45

### Programmatically (if you have a Fairseq model)
The code sample below uses an MT model trained with Fairseq v0.x (https://github.com/pytorch/fairseq).
However, it should be fairly easy to extend the code to another MT framework, by wrapping your model into a subclass of `translation_models.TranslationModel`.

```python
from pathlib import Path

from contrastive_evaluation import MTContrastiveEvaluationTask
from translation_models.fairseq_models import load_sota_model

# Warning: This will download a very large model from PyTorch Hub
model = load_sota_model()

testset_dir = Path("data") / "subj_verb_agreement.mt"
task = MTContrastiveEvaluationTask(
    src_path=testset_dir / "src.en",
    ref_path=testset_dir / "tgt.correct.de",
    contrastive_path=testset_dir / "tgt.incorrect.de",
)
result = task.evaluate(model)
print(result)
```

### From the command line (any MT system)
1. Use your MT system to score the translation variants in the data directory. For a given test set (e.g., `subj_verb_agreement.mt`), write the scores line by line into a file, similar to the *.scores files in https://github.com/rsennrich/lingeval97/tree/master/baselines. The first half of the file should be the scores for the correct translation variants (tgt.correct.de), the second half for the incorrect ones (tgt.incorrect.de).
2. Run the following command with the filepath as an argument:
```shell
python contrastive_evaluation.py \
  --testset-name subj_verb_agreement.mt \
  --scores-path myoutput.scores
```

## License
- Code: MIT License
- Data: Please refer to [OPUS](https://opus.nlpl.eu/) for the licenses of the `hypercorrect_genitive` data, and to the [WMT19 shared task website](https://opus.nlpl.eu/) for the license of the other data.

## Citation
```bibtex
@inproceedings{vamvas-etal-2021-limits,
    title = "On the Limits of Minimal Pairs in Contrastive Evaluation",
    author = "Vamvas, Jannis and
      Sennrich, Rico",
    booktitle = "Proceedings of the Fourth BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP",
    month = nov,
    year = "2021",
    publisher = "Association for Computational Linguistics"
}
```
