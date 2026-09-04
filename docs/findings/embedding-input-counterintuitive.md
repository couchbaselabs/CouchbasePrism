# The raw question embeds better than any distilled version of it

**Status:** measured, closed - kept the current behaviour
**Scale:** tested across all 143 answerable FinanceBench questions

## The intuition that turned out wrong

The evidence planner exists precisely because the raw user question is not a
good query: it is full of framing a source document never states back
("if we exclude the impact of M&A", "if conventional inventory management is
not meaningful"). BM25 already acts on that belief - the lexical channel
searches the planner's distilled `concept` and `content_anchors`, never the
raw question (`hybrid_search.py`).

The vector channel does the opposite: `vector_search.embed()` embeds the raw
question verbatim, unchanged since the beginning. That looked like an
oversight, not a decision - the same reasoning that motivated the lexical
side seemed to apply just as well to the vector side. It does not.

## What was measured

`eval/compare_embedding_input.py` holds the plan fixed per question (same
discipline as `compare_fusion.py`) and swaps only what text feeds `embed()`,
leaving the lexical channel and everything else untouched:

    A  raw question (current)
    B  the planner's concept alone ("quick ratio", "return on assets")
    C  question + concept appended
    D  concept + anchors merged into one deduped lexical bag (the same text
       BM25 already searches)
    E  concept + anchors joined as a natural, comma-like phrase

First pass, 8 3M questions only: **E won on every metric** (recall@5 0.62
vs. A's 0.46), which read as confirmation - distill the question, embed the
distillation, same story as the lexical side.

Full run, all 143 questions, told the opposite story:

    variant                          r@5   r@10  found    mean rank
    A  raw question (current)       0.72  0.88  131/143   2.7
    C  question + concept           0.71  0.88  131/143   2.7
    E  concept + anchors, natural   0.55  0.73  111/143   3.6
    D  merged lexical bag           0.52  0.70  105/143   3.7
    B  concept alone                0.33  0.53   82/143   4.2

The raw question is the best performer by a wide margin. Every distilled
variant is worse, and concept-alone is a severe regression - roughly 49 fewer
questions find their gold page in the top 10 at all, against the current
default.

The 8-question result was not wrong, it was unrepresentative: 3M's set
happened to concentrate narrative/attribution questions where the raw
question's framing genuinely is noise. That pattern inverts once the other
30 companies are included.

## A sharper, still-unresolved surprise

On `financebench_id_00807` ("Does 3M have a reasonably healthy liquidity
profile based on its quick ratio for Q2 of FY2023?"), none of the literal
words - "healthy", "liquidity", "profile", "quick ratio" - appear anywhere
in the gold table (a balance sheet: "Total current assets", "Total current
liabilities", "Cash and cash equivalents"). Embedding just the two words
that name the actual concept, "quick ratio", scored WORSE (rank 10) than the
full noisy sentence (rank 3).

Best available explanation, offered as a plausible reading of the data
rather than a proven mechanism - no interpretability work was done, only the
recall measurement above:

- Dense embedding models place text by topical neighbourhood, not literal
  term overlap. A full sentence carries several independent cues at once
  (liquidity, ratio, fiscal quarter, the entity) that individually correlate
  with balance-sheet-adjacent content from the model's training distribution,
  even where none of them is the table's own vocabulary. Several weak,
  redundant signals pulling the same direction outweigh zero exact matches.
- A two-word phrase is a much sparser input than a sentence. These models are
  trained and calibrated mostly on sentence/paragraph-scale text; a bare noun
  phrase's embedding is more easily pulled toward whatever commonly
  surrounds it in that training distribution (generic ratio-formula
  definitions, finance glossaries) than toward this specific filing's
  balance sheet. The full, syntactically normal sentence is better anchored.

## What this settles, and what it doesn't

Settled: the vector channel keeps embedding the raw question. This was tried,
measured, and reversed - not left alone by default. `compare_embedding_input.py`
stays in the repo as a kept negative result, the same way `hybrid_search.py`
keeps the reverted match_phrase-per-anchor attempt as a comment rather than
silently forgetting it was tried.

Not settled: WHY a full sentence anchors an embedding better than its own
distilled subject is inferred from behaviour, not verified against how the
embedding model was actually trained. Anyone revisiting this should treat the
mechanism as a hypothesis and the recall numbers as the only established fact.
