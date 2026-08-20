# Multi-period questions need more than one document

**Status:** open design question, not a bug
**Scale:** 14 of 143 answerable FinanceBench questions (~10%), the largest
remaining class of resolution failures

## The shape of it

Resolution returns exactly one `doc_name`, which then scopes retrieval, binding
and the trace. That is the right shape for most questions and the wrong shape
for these:

    What is Adobe's year-over-year change in unadjusted operating income
    from FY2015 to FY2016?

    What is the FY2017 - FY2019 3 year average of capex as a % of revenue
    for Activision Blizzard?

    Was there any drop in Cash & Cash equivalents between FY 2023 and
    Q2 of FY2024?

`period_from_question` takes the first year it finds, so the second period is
discarded silently. The answer is then computed from one filing and looks
plausible - which is worse than failing, because nothing in the trace says a
period went missing.

These are not mis-resolved. They are unanswerable in the current architecture.

## Why it is not a small fix

`resolved_doc` is a single string on the pipeline result, and everything
downstream assumes it:

- `hybrid_search` scopes both channels to one filename, inside `SEARCH()`.
- Fact binding assumes every fact comes from one document, and
  `validate_bindings` REJECTS a fact set drawn from more than one period - that
  check exists because mixing columns produces a number no arithmetic check can
  catch.
- The trace shows one document, one SQL statement, one evidence set.

So the period-mixing guard that makes single-period answers trustworthy is
exactly what a multi-period answer has to cross. That guard should not be
weakened; a multi-period answer needs facts tagged by period and compared
deliberately, not a relaxed check.

## Three options

**A. Resolve to a set, retrieve per document, bind per document.**
`resolve_for_question` returns a list. Retrieval runs once per document, and
bound facts carry their document and period. The mixed-period check becomes "one
value per fact per period" rather than "one period overall". Calculation gets a
notion of a value per period, so a year-over-year change is a first-class thing
rather than an accident.

Cost: touches resolution, retrieval scoping, binding validation, calculation and
the trace. It is the honest fix and it is not small.

**B. Detect and decline.**
Recognise that a question spans periods, and say so: "this question spans FY2015
and FY2016; PRISM answers from one filing at a time." Cheap, truthful, and
consistent with how the system already declines a verdict without an approved
policy. It converts 14 silent wrong answers into 14 explicit refusals.

Cost: nothing gained on the score. But the score stops being flattered by
answers that were never grounded in the periods asked for.

**C. Leave it.**
The current behaviour answers from whichever period was named first, with no
indication that another was requested. This is the only option that is
indefensible if someone asks what happened to the second year.

## Recommendation

**B now, A when multi-period answers are actually wanted.**

B is a few hours and removes a class of confidently wrong answers, which matters
more for a system whose entire claim is that it does not assert what it cannot
support. A is the real capability, and it is worth doing deliberately rather
than as a patch - particularly the part where bound facts carry their period, since
that is also what an approved formula spanning periods would need.

Doing A badly - by relaxing the mixed-period check so multi-document facts stop
being rejected - would trade 14 visible failures for an unknown number of
invisible ones. That check caught real errors.

## Measured context

Resolution accuracy on the 143 answerable questions, after the subject, fiscal
year, quarter and named-date fixes: 105/143 (73%). The remaining failures:

    15  wrong form          10-K vs earnings release for the same period;
                            mostly a judgment about which document type carries
                            the answer, not a keyword in the question
    14  multi-period        this document
     5  wrong subject       questions naming no subject at all
     4  wrong period

Retrieval given the correct document reaches recall@10 of 0.85, against 0.60
when the document is resolved - so resolution, not ranking, is where the
remaining recall lives.
