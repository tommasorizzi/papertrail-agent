# PaperTrail benchmark

## Scripted fixture run

Recorded from a clean Python 3.12.16 environment with networking disabled for
the run. Reproduce with `uv run --locked python scripts/run_eval.py`.

4 paired scripted fixtures · ScriptedModel · context cap 6

Corpus SHA-256 `2dfc8bca06e87e2e8b95600f89200f1822b15c2bf9405ebaa99c654af870a383`

Case-set SHA-256 `ff994383c2150ee50013abf2f24ab9b9abee21cae0e306f1fea9dc85d8ef2186`

| Metric | Single query | Graph |
| --- | ---: | ---: |
| Mean first-pass Recall@6 (cases) | 66.7% (3) | 83.3% (3) |
| Answered on fully answerable cases | 1/2 | 2/2 |
| Correct abstentions on unsupported cases | 1/1 | 1/1 |
| Correct partial statuses | 1/1 | 1/1 |
| False abstentions on fully answerable cases | 1/2 | 0/2 |
| False abstentions on partially answerable cases | 0/1 | 0/1 |
| Cases missing labeled passages after search | 1 | 0 |
| Cases with passages recovered on retry | 0 | 1 |
| Cases with retrieved labeled passages left uncited | 0 | 0 |
| Cases citing a labeled distractor | 0 | 0 |
| Errors | 0 | 0 |
| Median latency (ms) | 1.62 | 5.52 |
| Model calls per case | 1.00 | 2.50 |
| Total tokens reported | Unavailable | Unavailable |

Recall averages cases with relevant labels and a completed first search. Uncited passages and distractor citations flag cases for human review. Claim support requires a separate human rating.

## Example cases

### fixture-paraphrase

How was the comparison kept fair, and what was the cost of speeding up the transition from solid to liquid?

| Path | Status | Missing after first search | Missing after final search | Recovered on retry | Uncited retrieved labels | Calls |
| --- | --- | --- | --- | --- | --- | ---: |
| Single query | insufficient_evidence | pcm-copper-mesh-01, pcm-copper-mesh-02 | pcm-copper-mesh-01, pcm-copper-mesh-02 | None | None | 1 |
| Graph | answered | pcm-copper-mesh-02 | None | pcm-copper-mesh-02 | None | 3 |

### fixture-copper

Which passages describe a conductive insert in paraffin, and what trade-off do they report?

| Path | Status | Missing after first search | Missing after final search | Recovered on retry | Uncited retrieved labels | Calls |
| --- | --- | --- | --- | --- | --- | ---: |
| Single query | answered | None | None | None | None | 1 |
| Graph | answered | None | None | None | None | 2 |

## Live evaluation and limitations

The scripted cases use predetermined model outputs and six synthetic passages.
The latency figures measure the local workflow. The 16-case live holdout and
human claim-support ratings are pending. [Evaluation commands and review template](PaperTrail_Evaluation.md)

The next comparison is a paired live run on the fixed holdout, followed by
human review of each returned finding and its cited passage.
