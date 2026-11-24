# PaperTrail corpus provenance

The corpus contains six passages written as synthetic fixtures for
this project. They describe hypothetical experiments
with paraffin, aluminium fins, expanded graphite, and hydrated salt.
Each record carries a stable passage ID, title, section, source URL, source
terms, and passage text. [JSONL corpus](../data/corpus.jsonl)

The source terms identify the text as a PaperTrail fixture with no external
source. The URLs use the `example.invalid` placeholder domain. In the demo,
citations point to the bundled evidence cards and expandable full passages.

| Passage ID | Synthetic fixture | Section |
| --- | --- | --- |
| `pcm-copper-mesh-01` | Copper mesh in paraffin | Methods |
| `pcm-copper-mesh-02` | Copper mesh in paraffin | Results |
| `pcm-mesh-contact-01` | Wall contact in a copper mesh enclosure | Results |
| `pcm-aluminium-fins-01` | Aluminium fin spacing for a wax storage cell | Results |
| `pcm-graphite-01` | Expanded graphite composites for thermal storage | Material properties |
| `pcm-cycling-01` | Cycling a hydrated salt storage cell | Cycling observations |

The evaluation labels refer to these IDs. Run exports include a SHA-256
digest of the loaded corpus records and their order. [Evaluation labels](../data/eval_cases.jsonl) ·
[Evaluation method](PaperTrail_Evaluation.md)
