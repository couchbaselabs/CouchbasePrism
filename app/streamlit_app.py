"""PRISM showcase — the measured runtime, made inspectable.

Run from the repository root:

    streamlit run app/streamlit_app.py

The app deliberately calls ``prism.runtime.answer_question`` and
``eval.judge.score``. It does not maintain a friendlier demo-only pipeline.
"""
import html
import json
import os
import pathlib
import re
import sys
import time

from dataclasses import replace

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from eval import judge, phases  # noqa: E402
from eval.corpora import load as load_corpus  # noqa: E402
from prism import (  # noqa: E402
    catalog, config, couchbase_io, dictionary, retrieval, runtime, trace,
)
from prism import initialize as prism_initialize  # noqa: E402
from prism import s3_upload  # noqa: E402


st.set_page_config(
    page_title="PRISM · Governed retrieval",
    page_icon=":material/account_tree:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit's wide layout is right for the benchmark table and trace, but its
# default side padding is generous. Keep a narrow responsive gutter while
# ensuring that long diagnostic content scrolls inside its element instead of
# widening the browser viewport.
st.html("""
<style>
  /* --- MAIN PANEL --- */
  [data-testid="stMainBlockContainer"] {
    max-width: 90% !important;
    margin-left: 0.5rem !important;
    margin-right: auto !important;

    /* Streamlit's own header is fixed-position, z-index 999990, and really is
       3.75rem tall (measured: 48.75px at this app's 13px root font-size) -
       ANY padding-top below that lets content slide underneath it rather
       than sit below it. 1.5rem was invisible as a bug until today: nothing
       had ever been the literal first element in the main body before the
       tabs added for the doc workbench, so nothing had tested this edge
       before. Verified live (not by eye - a partial overlap is invisible in
       a screenshot): getBoundingClientRect() on the header vs the first
       child, not guessed. Keep this at or above 3.75rem. */
    padding-top: 3.9rem !important;

    padding-left: clamp(0.75rem, 1.4vw, 1.5rem) !important;
    padding-right: clamp(0.75rem, 1.4vw, 1.5rem) !important;
  }

  /* --- SIDEBAR --- */
  /* Reduce space inside the sidebar content area */
  [data-testid="stSidebarUserContent"] {
    padding-top: 1rem !important; /* Adjust 1rem as needed */
  }

  /* Tighten space above the sidebar header/collapse toggle button */
  [data-testid="stSidebarHeader"] {
    padding-top: 0.5rem !important;
    padding-bottom: 0 !important;
    height: auto !important;
  }

  /* Prevent horizontal browser scroll */
  [data-testid="stMain"] {
    overflow-x: clip !important;
  }
  [data-testid="stCode"] pre {
    max-width: 100% !important;
    overflow-x: auto !important;
  }

  /* Elevation is reserved for the three primary surfaces. Trace expanders
     retain their flat treatment so the page still has a clear hierarchy. */
  .st-key-prompt-card,
  .st-key-benchmark-summary,
  .st-key-answer-card {
    border-radius: 10px;
    box-shadow: 0 1px 2px rgba(23, 43, 45, 0.06),
                0 7px 20px rgba(23, 43, 45, 0.07);
  }
</style>
""")
MODEL_OPTIONS = ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4", "gpt-5.5",
                 "gpt-4.1-mini", "gpt-4o-mini"]
# Roles, not call sites - see prism.config.STAGE_ROLE. Listed in the order
# they first appear in the pipeline trace below (steps 1/2/4/5/7/8), not
# alphabetically or by cost. "Classification" used to sit third in an
# unordered list, right after "Binding + answer", and looked like it ran
# after the answer stage. It doesn't: per-question document resolution is
# deterministic code, not an LLM call, and catalog field extraction is an
# ingestion-time job - this role rarely appears in a single question's trace
# at all. Descriptions below cite the trace step number so the two panels
# read as one pipeline instead of two different orderings.
MODEL_ROLES = [
    ("utility", "Catalog matching", "gpt-5.4",
     "Step 1 (Filter), mostly at ingestion: extracted company, form and period "
     "across 354 documents without a failure. Also covers anchor repair, off "
     "by default. Cheap and high volume."),
    ("planner", "Planner", "gpt-5.4",
     "Steps 2 (Plan) and 4 (Govern): the evidence plan, and - only when "
     "nothing approved matches - candidate calculation conventions. Proposes "
     "structure it cannot verify."),
    ("answer", "Bind + answer", "gpt-5.4",
     "Steps 5 (Bind) and 7 (Answer): binds facts to rows and columns, then "
     "writes the answer. Errors here are wrong numbers in front of a reader."),
    ("judge", "Evaluation judge", "gpt-5.4",
     "Step 8 (Evaluate): grades the answer against the benchmark reference. "
     "Evaluation only, never part of an answer path. Keep it OFF the answer "
     "model: a model grading its own output is how gpt-5.5 came to look "
     "worse than gpt-5.4."),
]
PHASE_HELP = {
    "1-vector": "Textbook RAG · kNN across the whole corpus, no scoping",
    "2-catalog": "Adds catalog-resolved document scoping before similarity",
    "3-hybrid": "Preferred · BM25 + content anchors + kNN via SEARCH(), "
                "with binding, deterministic calculation and governance",
}
FUSION_LABELS = {"score": "Additive (default)", "native-rrf": "Native RRF",
                 "native-rsf": "Native RSF", "rrf": "RRF in code"}
FUSION_HELP = {
    "score": "Bleve's default: weighted addition of the lexical and vector scores. "
             "Sensitive to the two scores being on different scales, so a "
             "lexical-only hit can lose to every vector hit - but measured equal to "
             "RRF and RSF on recall over 143 questions.",
    "native-rrf": "Server-side reciprocal rank fusion, 1/(k + rank), one SEARCH(). "
                  "Needs Couchbase 8.1 - on 8.0.1 the score field parsed and was "
                  "silently ignored. Channel weights come from each query's boost.",
    "native-rsf": "Relative score fusion: min-max normalise each channel into "
                  "[0,1], then add with the query boosts as weights. Keeps score "
                  "magnitude, but one outlier skews the normalisation.",
    "rrf": "Two SEARCH channels unioned in one statement, fused in application "
           "code. Works on any version, and the only option that reports each "
           "channel's rank and contribution per chunk.",
}
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PRISM_MARK = REPO_ROOT / "app" / "assets" / "prism-mark.png"


def tuning_panel(fusion: str) -> dict:
    """Retrieval dials, each with a one-line statement of what it is.

    Every control returns None when untouched, so a run that touches no dial is
    identical to one from before this panel existed.
    """
    out = {}
    with st.expander("Tuning surface", icon=":material/tune:"):
        if fusion in ("native-rrf", "rrf"):
            out["rank_constant"] = st.slider(
                "Rank constant", 1, 120, config.RRF_K, key="dial_rank_constant")
            st.caption("The k in 1/(k + rank). Low values reward being first in a "
                       "channel; high values reward appearing in both.")

        out["window_size"] = st.slider(
            "Window size", 10, 400, config.NATIVE_WINDOW_SIZE, step=10,
            key="dial_window")
        st.caption("How many results from each channel are considered for fusion. "
                   "Must be at least the evidence budget.")

        left, right = st.columns(2)
        out["bm25_weight"] = left.number_input("Lexical weight", 0.0, 10.0, 1.0, 0.10,
                                               key="dial_bm25")
        out["vector_weight"] = right.number_input("Vector weight", 0.0, 10.0, 1.0, 0.10,
                                                  key="dial_vector")
        st.caption("Each channel's relative importance, written into the statement "
                   "as that query's boost. Equal weights mean neither is favoured.")

        out["knn_k"] = st.slider("Vector candidate depth", 10, 500,
                                 config.KNN_CANDIDATES, step=10, key="dial_knn")
        st.caption("How many nearest neighbours the vector channel offers up. A "
                   "chunk outside this depth cannot be fused, however well it "
                   "matches lexically.")

        out["top_k"] = st.slider("Evidence budget", 3, 25, config.TOP_K,
                                 key="dial_topk")
        st.caption("How many chunks reach the answer stage. Smaller budgets make "
                   "rank order matter more.")
    return {k: v for k, v in out.items() if v is not None}


@st.cache_data(ttl=60, show_spinner=False)
def load_catalog():
    return catalog.load_all()


@st.cache_data(show_spinner=False)
def load_questions():
    # ftsprism, not financebench - the demo runs on PRISM's own open corpus
    # now; FinanceBench stays available for eval.run_benchmark's internal
    # comparison, but has no place in the public-facing "Ask a question" tab.
    return load_corpus("ftsprism").questions()


def all_sectors() -> dict:
    """{doc_name: gics_sector} merged across every corpus that's actually
    present locally - not just financebench's. financebench/ is gitignored
    and cloned separately, so a self-contained checkout (ftsprism's whole
    point) legitimately might not have it; sectors() there opens a file
    unconditionally and would raise on a clean checkout that never cloned
    it. ftsprism's own sectors always merge in on top since it ships in the
    repo itself."""
    merged = {}
    for name in ("financebench", "ftsprism"):
        try:
            merged.update(load_corpus(name).sectors())
        except OSError:
            pass
    return merged


def company_catalog(catalog_docs: list, company: str) -> list:
    """Temporary eval adapter until catalog resolution is entity-aware."""
    prefix = company.upper().replace(" ", "") + "_"
    matched = [d for d in catalog_docs
               if (d.get("doc_name") or "").upper().replace(" ", "").startswith(prefix)]
    return matched or catalog_docs


def fetch_chunks(doc_name: str, page: int = None, search_terms: str = None,
                 limit: int = 50, source_filename: str = None) -> list:
    """Chunks for one document, JSON-ready. The embedding vector is stripped
    in Python rather than excluded in SQL++ - nothing elsewhere in this
    codebase relies on an EXCLUDE clause, and this keeps the query portable
    rather than depending on a possibly version-gated extension.

    The whole filter goes through the Search Vector Index via one SEARCH(),
    not plain N1QL WHERE predicates - the collection has no GSI on filename or
    page-number, so a plain `d.xmeta-data.filename = $filename` predicate
    fell back to a primary index scan. Confirmed against the live index
    definition (GET /api/bucket/{bucket}/scope/{scope}/index/ftsPrism)
    that both fields ARE mapped: xmeta-data.filename as text/keyword analyzer
    (an exact-match `match`, same as the retrieval path already uses),
    meta-data.page-number as a `number` field - FTS has no numeric term query,
    so an exact page is a min/max range collapsed to one point.

    source_filename should be the caller's already-resolved catalog entry
    field - same reasoning as the runtime pipeline's own source_filename
    threading (prism/runtime/pipeline.py): recomputing config.source_filename
    fresh on every call means AWS_BUCKET/AWS_FOLDER have to stay correct
    forever, not just once. Falls back to recomputing only when the caller
    has no catalog entry to read it from.

    search_terms scopes to the WHOLE document, not one page - `page` is
    ignored entirely when terms are given, not merely de-prioritized, since
    finding where a term appears is the point and a page constraint would
    defeat it. Same text-to-embed field and match/OR shape the retrieval
    path's BM25 leg already uses (the index's own scoring_model is "bm25",
    confirmed from its live definition), not a raw LIKE. A #...# span inside
    search_terms is a forced exact-phrase match, same syntax and same
    extract_phrase_terms() as the runtime pipeline's Path A (prism/retrieval/
    planner.py) - added as its own match_phrase disjunct alongside the usual
    bag-of-words match, not instead of it.

    A document-wide term search has no natural bound the way a single page
    does, so this caps at `limit` - callers should tell the user when the
    result is exactly at the cap, since that's the signal more exist.
    """
    conjuncts = ['{"field": "xmeta-data.filename", "match": $filename}']
    params = {"$filename": source_filename or config.source_filename(doc_name)}
    if search_terms:
        # A forced phrase is a DISJUNCT alongside the bag-of-words match, not
        # an extra required conjunct - "either finds it" is the point, same
        # as Path A's own lexical leg (prism/retrieval/hybrid_search.py). A
        # top-level conjunct here would instead require BOTH to match, which
        # is a much narrower (and wrong) search than what #...# is for.
        cleaned_terms, phrases = retrieval.extract_phrase_terms(search_terms)
        term_disjuncts = ['{"match": $terms, "field": "text-to-embed", '
                         '"operator": "or"}']
        params["$terms"] = cleaned_terms
        for i, phrase in enumerate(phrases):
            key = f"$phrase_{i}"
            term_disjuncts.append(f'{{"match_phrase": {key}, "field": "text-to-embed"}}')
            params[key] = phrase
        conjuncts.append(f'{{"disjuncts": [{", ".join(term_disjuncts)}]}}')
        # A term search should surface the best match first, not page order -
        # on 3M's 2022 10-K, "current assets" in page order buries the
        # Working Capital table (page 38, which nets current assets against
        # current liabilities - the strongest possible match) under a dozen
        # boilerplate risk-factor pages that only happen to contain "current"
        # and "assets" separately. SEARCH_SCORE() puts that table first.
        order_by = "SEARCH_SCORE() DESC"
    else:
        conjuncts.append(
            '{"field": "meta-data.page-number", "min": $page, "max": $page, '
            '"inclusive_min": true, "inclusive_max": true}')
        params["$page"] = page
        # No relevance to rank by here - an exact filename+page filter is a
        # deterministic match, not a search. Reading order within the page.
        order_by = "d.`meta-data`.type, d.`element-id`"
    rows = couchbase_io.query(
        "SELECT META(d).id AS _id, d.*, SEARCH_SCORE() AS _score "
        f"FROM `{config.BUCKET}`.`{config.SCOPE}`.`{config.DOCS_COLLECTION}` AS d "
        f'WHERE SEARCH(d, {{"query": {{"conjuncts": [{", ".join(conjuncts)}]}}}}, '
        f'{{"index": "{config.FTS_DOCS_INDEX}"}}) '
        f"ORDER BY {order_by} "
        f"LIMIT {int(limit)}",
        params,
    )
    for row in rows:
        row.pop("text-embedding", None)
    return rows


def highlight_terms(text: str, terms: str) -> str:
    """text-to-embed with every matched search word wrapped in <mark>, so a
    hit reads as a hit instead of a wall of text to re-search by eye.

    Escapes the base text first, exactly like _md_bold() elsewhere in this
    file - then escapes each search word too before building the regex, so a
    word containing an HTML special character still matches literally against
    the now-escaped text rather than silently failing to match or reopening
    the injection risk _md_bold was written to close.
    """
    escaped_text = html.escape(text or "")
    words = [html.escape(w) for w in re.split(r"\s+", (terms or "").strip()) if w]
    if not words:
        return f'<div style="white-space:pre-wrap">{escaped_text}</div>'
    pattern = re.compile("(" + "|".join(re.escape(w) for w in words) + ")", re.IGNORECASE)
    highlighted = pattern.sub(
        r'<mark style="background:#fde68a;color:#1a1a1a;padding:0 2px;'
        r'border-radius:2px">\1</mark>',
        escaped_text)
    return f'<div style="white-space:pre-wrap;line-height:1.5">{highlighted}</div>'


def run_initialize(status, sectors: dict) -> dict:
    """Runs prism_initialize.run(), rendering progress into an already-open
    st.status container. A real function, not inlined into the button's
    click handler: Streamlit's script body executes at MODULE scope, and
    nonlocal - which on_step/on_catalog_progress need to update the catalog
    progress bar across calls - has no enclosing FUNCTION scope to bind to
    there. Caught by actually loading the app, not by ast.parse: it checks
    syntax, not scoping validity, so a SyntaxError this shape only surfaces
    at import/exec time.
    """
    catalog_bar, catalog_line = None, None
    index_bar, index_line = None, None

    def on_step(i, total, name, step_status, detail=None):
        nonlocal catalog_bar, catalog_line, index_bar, index_line
        if step_status == "running":
            status.write(f"[{i}/{total}] {name}…")
            if name.startswith("Rebuild the catalog"):
                catalog_bar = st.progress(0.0)
                catalog_line = st.empty()
            elif name.startswith("Wait for the search index"):
                index_bar = st.progress(0.0)
                index_line = st.empty()
        elif step_status == "error":
            status.write(f"[{i}/{total}] {name}: ERROR {detail}")

    def on_catalog_progress(i, total, doc_name, result):
        if catalog_bar:
            catalog_bar.progress(i / total)
            catalog_line.caption(
                f"[{i}/{total}] {doc_name}"
                + ("" if result["ok"] else f" — ERROR: {result['error']}"))

    def on_index_progress(count, target):
        if index_bar:
            index_bar.progress(min(count / target, 1.0) if target else 1.0)
            index_line.caption(f"{count}/{target} documents reindexed")

    # Deliberately model=None: an admin operation, independent of whatever the
    # user picks per-role for answering a question elsewhere in the sidebar
    # (that dict doesn't exist yet at this point in the script anyway). None
    # resolves to config.MODEL_UTILITY, same as manage.py initialize.
    return prism_initialize.run(model=None, sectors=sectors, on_step=on_step,
                                on_catalog_progress=on_catalog_progress,
                                on_index_progress=on_index_progress)


def trace_stats(events: list) -> dict:
    llm_events = [e for e in events if e.get("type") == "llm_call"]
    sql_events = [e for e in events if e.get("type") == "couchbase_query"]
    embedding_events = [e for e in events if e.get("type") == "embedding_call"]
    usage = [e.get("usage") or {} for e in llm_events + embedding_events]
    summary = next((e for e in reversed(events)
                    if e.get("type") == "trace_summary"), {})
    return {
        "elapsed_ms": summary.get("elapsed_ms", 0),
        "llm_calls": len(llm_events),
        "sql_calls": len(sql_events),
        "embedding_calls": len(embedding_events),
        "prompt_tokens": sum(u.get("prompt_tokens", u.get("input_tokens", 0)) or 0
                             for u in usage),
        "completion_tokens": sum(u.get("completion_tokens", u.get("output_tokens", 0)) or 0
                                 for u in usage),
        "total_tokens": sum(u.get("total_tokens", 0) or 0 for u in usage),
    }


def run_one(question: dict, catalog_docs: list, dictionary_data: dict,
            phase: str, model: str, fusion: str = None, tuning: dict = None) -> dict:
    started = time.perf_counter()
    overrides = dict(tuning or {})
    if fusion:
        overrides["fusion"] = fusion
    options = replace(phases.get(phase), **overrides) if overrides else phases.get(phase)
    with trace.capture() as events:
        result = runtime.answer_question(
            question["question"], company_catalog(catalog_docs, question["company"]),
            dictionary_data, options=options, model=model)
        # A custom question has no FinanceBench reference to judge against -
        # question["expected_answer"] is display text for the UI, not data a
        # judge call should ever see.
        verdict = ({"passed": None, "method": "unscored",
                   "comment": "Custom question — no reference answer to score against."}
                  if question["id"] == "custom" else
                  judge.score(question["question"], question["expected_answer"],
                             result["answer"], model=model))
    stats = trace_stats(events)
    stats["wall_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return {"question": question, "result": result, "verdict": verdict,
            "events": events, "stats": stats}


def section(title: str, caption: str = None, icon: str = None):
    label = f"{icon} {title}" if icon else title
    st.subheader(label)
    if caption:
        st.caption(caption)


def status_badge(passed):
    if passed is True:
        st.badge("Converged", icon=":material/check:", color="green")
    elif passed is False:
        st.badge("Diverged", icon=":material/close:", color="red")
    else:
        st.badge("Not scored", icon=":material/pending:", color="gray")


def llm_role(event: dict) -> str:
    """Each call site tags itself via llm.chat_*(stage=...). The fallback below
    only covers traces recorded before that tag existed - sniffing prompt text
    was the original mechanism and it silently mislabelled every call the moment
    a prompt was reworded."""
    if event.get("stage"):
        return event["stage"]
    messages = (event.get("request") or {}).get("messages") or []
    text = "\n".join(str(m.get("content", "")) for m in messages).lower()
    if "planning the source evidence" in text:
        return "planner"
    if "binding named facts" in text:
        return "binder"
    if "proposing possible calculation methods" in text:
        return "candidate"
    if "grading whether a candidate answer" in text:
        return "judge"
    return "answer"


def condense_excerpts(text: str) -> str:
    """Replace each excerpt body with a size marker, keeping the page and title
    headers. The answer prompt carries ten chunks of full document text, which
    buries the instructions being inspected - and the bodies are already
    readable, per chunk, on the Evidence tab.

    Display only. The prompt sent to the model is untouched: dropping the
    excerpt bodies from the real request would leave nothing to answer from.
    """
    def shrink(match):
        return f"{match.group(1)}<{len(match.group(2)):,} chars - see Evidence tab>"
    return re.sub(r"(\bContent: )(.*?)(?=\n\n---\n\n|\n\nQUESTION:|\Z)",
                  shrink, text, flags=re.S)


def show_llm_exchange(event: dict, label: str):
    """The prompt is the interesting part of an LLM call, so it is rendered as
    readable text. Collapsed JSON technically contained it, but as one escaped
    string per message - unreadable exactly when it matters, which is while
    working out why a prompt produced the plan it did."""
    request = event.get("request") or {}
    usage = event.get("usage") or {}
    elapsed = event.get("elapsed_ms", 0)
    st.caption(f"{label} · {request.get('model', 'model')} · "
               f"{usage.get('total_tokens', '—')} tokens · {elapsed:,.0f} ms")
    prompt_tab, response_tab, raw_tab = st.tabs(["Prompt", "Response", "Raw request"])
    with prompt_tab:
        st.caption("Excerpt bodies are elided here - the full text of every "
                   "retrieved chunk is on the Evidence tab. Raw request has the "
                   "prompt exactly as sent.")
        for message in request.get("messages") or []:
            st.markdown(f"**{message.get('role', 'message')}**")
            st.code(condense_excerpts(str(message.get("content", ""))),
                    wrap_lines=True)
    with response_tab:
        raw = event.get("response") or event.get("error") or ""
        try:
            st.json(json.loads(raw), expanded=True)
        except (ValueError, TypeError):
            # Answer synthesis returns prose, not JSON.
            st.code(raw, wrap_lines=True)
    with raw_tab:
        st.json(request, expanded=False)


def sql_kind(event: dict) -> str:
    """Name each statement by what it actually does. Generic 'SQL++ 1 / 2'
    labels made it impossible to tell at a glance whether the FTS index was
    being used at all."""
    statement = event.get("statement", "")
    if "`catalog`" in statement:
        return "Catalog lookup"
    if "matched_anchors" in statement:
        return "Content anchors · exact row-label match"
    if "SEARCH(" in statement:
        # All three retrieval shapes are now SEARCH() against the same Search
        # Vector Index, so the label has to come from the query object rather
        # than from the function name. Order matters: the union carries both.
        if "UNION ALL" in statement:
            return ("Two channels, one statement · BM25 + kNN unioned, "
                    "fused by reciprocal rank")
        if "match_phrase" in statement or "disjuncts" in statement:
            return "Hybrid SEARCH() · BM25 + kNN fused by the Search service"
        return "Vector kNN only · Search Vector Index"
    return "SQL++"


PARAM_TOKEN = re.compile(r"\$[a-zA-Z_][a-zA-Z_0-9]*")


def pretty_sql(statement: str) -> str:
    """Expand the JSON objects embedded in SEARCH() so the query object is
    readable. Everything hard about these statements is WHERE the predicates
    land, and that is invisible when the whole search object is one long line.

    Named parameters are not valid JSON, so they are quoted before parsing and
    unquoted afterwards.
    """
    out, i = [], 0
    while i < len(statement):
        ch = statement[i]
        if ch == "{":
            depth, j = 0, i
            while j < len(statement):
                if statement[j] == "{":
                    depth += 1
                elif statement[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            blob = statement[i:j + 1]
            quoted = PARAM_TOKEN.sub(lambda m: f'"{m.group(0)}"', blob)
            try:
                parsed = json.loads(quoted)
            except ValueError:
                out.append(blob)
            else:
                # Unquote the parameters again so the statement reads as SQL++.
                out.append(re.sub(r'"(\$[a-zA-Z_][a-zA-Z_0-9]*)"', r"\1",
                                  json.dumps(parsed, indent=2)))
            i = j + 1
            continue
        out.append(ch)
        i += 1
    text = "".join(out)
    return "\n".join(line.rstrip() for line in text.splitlines() if line.strip())


def show_sql(event: dict, label: str):
    elapsed = event.get("elapsed_ms", 0)
    st.caption(f"{label} · {event.get('row_count', 0)} rows · {elapsed:,.0f} ms")
    st.code(pretty_sql(event.get("statement", "")), language="sql", wrap_lines=True)
    with st.popover("Parameters and query metrics", icon=":material/query_stats:"):
        st.markdown("**Parameters**")
        st.json(event.get("params") or {}, expanded=True)
        if event.get("metrics"):
            st.markdown("**Metrics**")
            st.json(event["metrics"], expanded=True)


def render_pipeline_trace(run: dict):
    q, result, events = run["question"], run["result"], run["events"]
    llm_events = [e for e in events if e.get("type") == "llm_call"]
    llms = {role: [e for e in llm_events if llm_role(e) == role]
            for role in ("planner", "candidate", "binder", "answer", "judge",
                        "catalog")}  # "catalog" = fts_resolver.llm_resolve(),
                                     # present only when Document resolution
                                     # = AI; classify_cover() also tags
                                     # "catalog" but runs during Initialize,
                                     # never inside an answer_question() trace
    # Every call must surface somewhere. When classification silently dropped
    # calls into the wrong bucket, the trace lost the planner and binder
    # prompts entirely and looked merely sparse rather than broken.
    classified = {id(e) for bucket in llms.values() for e in bucket}
    llms["unclassified"] = [e for e in llm_events if id(e) not in classified]
    sql_events = [e for e in events if e.get("type") == "couchbase_query"]
    embed_events = [e for e in events if e.get("type") == "embedding_call"]
    plan = result["plan"]
    options = result["options"]
    # These are independent switches, so describe every one that is on. An
    # if/elif chain here reported only the first and made an active BM25 leg
    # look like it was never running.
    active = ([" content anchors"] if options.anchors else []) \
        + (["BM25 lexical"] if options.bm25 else []) + ["vector kNN"]
    retrieval_mode = " + ".join(part.strip() for part in active)

    with st.expander("1 · Filter — resolve the governed document scope", expanded=True,
                     icon=":material/filter_alt:"):
        st.markdown("**In** · user question + company-filtered catalog  ")
        if options.catalog_filter:
            st.markdown("**Work** · one SEARCH() against the catalog's own index narrows to "
                        "a shortlist, then a cheap LLM call picks from it (or declines) - "
                        "no full catalog scan, no regex parsing the question's phrasing.  ")
            for number, event in enumerate(llms["catalog"], 1):
                show_llm_exchange(event, "Catalog resolver"
                                  + (f" · call {number}" if len(llms["catalog"]) > 1 else ""))
        else:
            st.markdown("**Work** · catalog filtering is intentionally disabled in this "
                        "baseline phase, so retrieval searches the corpus.  ")
        st.markdown("**Out** · document key used as the retrieval predicate")
        if result["resolved_doc"]:
            st.success(f"`{result['resolved_doc']}`")
        else:
            st.warning("No document predicate · phase 1 baseline")

    with st.expander("2 · Plan — identify the evidence the answer requires", expanded=True,
                     icon=":material/route:"):
        st.markdown("**In** · question  ")
        st.markdown("**Work** · the planner identifies the concept, answer type, facts, "
                    "artifact types and verbatim content anchors. No dictionary is required.  ")
        for number, event in enumerate(llms["planner"], 1):
            show_llm_exchange(event, "Evidence planner"
                              + (f" · call {number}" if len(llms["planner"]) > 1 else ""))
        st.markdown("**Out**")
        st.json(plan, expanded=True)

    with st.expander("3 · Retrieve — anchors + semantic search inside the document",
                     expanded=True, icon=":material/manage_search:"):
        st.markdown("**In** · resolved document key + content anchors + question embedding  ")
        st.markdown(f"**Active strategy** · `{retrieval_mode}`  ")
        work = []
        if options.anchors:
            work.append("exact content-anchor matches identify structurally relevant chunks, "
                        "ranked by anchor rarity rather than hit count")
        if options.bm25:
            work.append("the question goes to the **Search Vector Index**, which combines "
                        "**BM25** lexical scoring with vector similarity in a single `SEARCH()` "
                        "call")
        else:
            work.append("vector similarity search fills the evidence budget")
        if options.catalog_filter:
            work.append("everything is scoped to the resolved document")
        st.markdown("**Work** · " + "; ".join(work)
                    + (". Results are deduplicated with anchor hits first.  "
                       if options.anchors else ".  "))
        for event in embed_events:
            st.caption(f"Embedding · {event.get('dimensions', '—')} dimensions · "
                       f"{event.get('elapsed_ms', 0):,.0f} ms")
            st.json(event.get("request") or {}, expanded=False)
        for number, event in enumerate(sql_events, 1):
            show_sql(event, f"{number} · {sql_kind(event)}")
        st.markdown(f"**Out** · {len(result['chunks'])} evidence chunks")

    with st.expander("4 · Govern — apply approved semantics or surface ambiguity",
                     expanded=True, icon=":material/policy:"):
        st.markdown("**In** · planner concept + scoped dictionary  ")
        if result["dictionary_entry"]:
            st.success(f"Approved metric: `{result['dictionary_entry']}`")
        else:
            st.warning("No approved metric matched. Candidate interpretations may be proposed.")
        st.write("Interpretation policy:", "Approved" if result["has_policy"] else "None")
        for number, event in enumerate(llms["candidate"], 1):
            show_llm_exchange(event, "Candidate-formula proposal"
                              + (f" · call {number}" if len(llms["candidate"]) > 1 else ""))
        if result["candidates"]:
            st.json(result["candidates"], expanded=True)

    with st.expander("5 · Bind — connect each fact to a row, column and source",
                     expanded=bool(result["bound_facts"]), icon=":material/link:"):
        st.markdown("**In** · required fact names + retrieved evidence  ")
        st.markdown("**Work** · bind each value to entity, units, period, printed row and "
                    "source page; reject values absent from the retrieved text or mixed periods.  ")
        for number, event in enumerate(llms["binder"], 1):
            show_llm_exchange(event, "Fact binder"
                              + (f" · call {number}" if len(llms["binder"]) > 1 else ""))
        if result["bound_facts"]:
            st.dataframe([{
                "Grounded": bool(f.get("grounded")), "Fact": f.get("name"),
                "Value": f.get("value"), "Units": f.get("units"),
                "Row": f.get("row_label"), "Column": f.get("period"),
                "Page": f.get("source_page"), "Issues": "; ".join(f.get("binding_issues") or [])
            } for f in result["bound_facts"]], hide_index=True, width="stretch")
        else:
            st.caption("This answer path did not require fact binding.")

    with st.expander("6 · Calculate and validate — deterministic arithmetic, governed verdict",
                     expanded=True, icon=":material/calculate:"):
        st.markdown("**In** · grounded facts + approved formula, or labeled candidate formulas  ")
        calc = result["calculation"]
        if calc["computed"]:
            for item in calc["computed"]:
                st.metric(item["label"], f"{item['value']:.6g}", border=True)
                st.code(item["formula"])
        else:
            st.caption("No deterministic calculation was needed or possible.")
        if calc["errors"]:
            st.error("\n".join(e["error"] for e in calc["errors"]))
        if result["conclusion"]:
            st.markdown("**Out · validation decision**")
            st.json(result["conclusion"], expanded=True)

    with st.expander("7 · Answer — synthesize prose around grounded evidence", expanded=True,
                     icon=":material/chat:"):
        st.markdown("**In** · evidence chunks + already-computed values + validation state  ")
        for number, event in enumerate(llms["answer"], 1):
            show_llm_exchange(event, "Answer synthesis"
                              + (f" · call {number}" if len(llms["answer"]) > 1 else ""))
        st.markdown("**Out**")
        st.info(result["answer"], icon=":material/auto_awesome:")

    with st.expander("8 · Evaluate — compare with the benchmark convention",
                     icon=":material/fact_check:"):
        st.caption("Evaluation-only. This is not part of a production answer path.")
        for number, event in enumerate(llms["judge"], 1):
            show_llm_exchange(event, "Gold-answer judge"
                              + (f" · call {number}" if len(llms["judge"]) > 1 else ""))
        status_badge(run["verdict"]["passed"])
        st.write(run["verdict"].get("comment", ""))

    if llms["unclassified"]:
        with st.expander(f"Unclassified LLM calls ({len(llms['unclassified'])})",
                         icon=":material/help:"):
            st.caption("These calls could not be attributed to a pipeline stage. A trace "
                       "recorded before stage tagging existed will land here; so will a "
                       "call site that forgot to tag itself.")
            for number, event in enumerate(llms["unclassified"], 1):
                show_llm_exchange(event, f"Untagged call {number}")


PASS_COLOR = "#1a7f37"
FAIL_COLOR = "#cf222e"

# st.dataframe truncates every cell and makes the reader click to read an
# answer, which is the wrong shape for a comparison table where the answers ARE
# the content. This renders a fully expanded table instead: nothing clipped,
# nothing scrollable, every row readable top to bottom.
RESULTS_TABLE_CSS = """
<style>
.prism-results { width: 100%; border-collapse: collapse; font-size: 0.86rem;
                 line-height: 1.45; table-layout: fixed; }
.prism-results-shell { width: 100%; border: 1px solid rgba(128,128,128,0.22);
                       border-radius: 10px; background: #fff;
                       box-shadow: 0 1px 2px rgba(23,43,45,0.05),
                                   0 8px 24px rgba(23,43,45,0.07); }
.prism-results th { text-align: left; font-weight: 600; padding: 0.6rem 0.7rem;
                    border-bottom: 2px solid rgba(128,128,128,0.35);
                    vertical-align: bottom; white-space: nowrap; position: sticky;
                    top: 0; z-index: 1; background: #f3f7f7; }
.prism-results th:first-child { border-top-left-radius: 9px; }
.prism-results th:last-child { border-top-right-radius: 9px; }
.prism-results td { padding: 0.7rem; vertical-align: top;
                    border-bottom: 1px solid rgba(128,128,128,0.18);
                    overflow-wrap: anywhere; }
.prism-results tbody tr:last-child td { border-bottom: 0; }
.prism-results tbody tr:last-child td:first-child { border-bottom-left-radius: 9px; }
.prism-results tbody tr:last-child td:last-child { border-bottom-right-radius: 9px; }
.prism-results tr:nth-child(even) td { background: rgba(128,128,128,0.05); }
.prism-results tbody tr:hover td { background: rgba(11,107,105,0.075); }
/* nowrap matters: the cell sets overflow-wrap:anywhere for long prose, which
   without this breaks the pill itself into "PAS / S". */
.prism-pill { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
              color: #fff; font-weight: 700; font-size: 0.75rem; letter-spacing: .03em;
              white-space: nowrap; overflow-wrap: normal; }
.prism-num { text-align: right; font-variant-numeric: tabular-nums;
             white-space: nowrap; color: rgba(128,128,128,0.95); }
.prism-doc { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
             font-size: 0.76rem; overflow-wrap: anywhere; }
</style>
"""

RESULT_COLUMNS = [
    ("Status", "9%"), ("Question", "19%"), ("Gold answer", "21%"),
    ("PRISM answer", "25%"), ("Why", "15%"), ("Document", "11%"),
]


def _md_bold(text: str) -> str:
    """Escape first, then turn **bold** into <strong> - the one bit of the LLM's
    own markdown this table renders. st.html() does not run a markdown parser
    (that was the source of the leaked-tag bug this replaced), so **emphasis**
    from an answer or judge comment would otherwise show as literal asterisks
    instead of being dropped silently or corrupting tags."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(text))


def results_table_html(runs: list) -> str:
    header = "".join(f"<th>{html.escape(name)}</th>" for name, _ in RESULT_COLUMNS)
    cols = "".join(f'<col style="width:{width}">' for _, width in RESULT_COLUMNS)
    body = []
    for run in runs:
        q, result, verdict = run["question"], run["result"], run["verdict"]
        passed = verdict["passed"]
        colour = PASS_COLOR if passed else FAIL_COLOR
        label = "PASS" if passed else "FAIL"
        cells = [
            f'<td><span class="prism-pill" style="background:{colour}">{label}</span>'
            f'<div class="prism-num" style="text-align:left;margin-top:.35rem">'
            f'{html.escape(q["id"].replace("financebench_id_", "#"))}</div></td>',
            f'<td>{_md_bold(q["question"])}</td>',
            f'<td>{_md_bold(q["expected_answer"])}</td>',
            f'<td>{_md_bold(result["answer"])}</td>',
            f'<td>{_md_bold(verdict.get("comment") or "")}</td>',
            f'<td class="prism-doc">{html.escape(result["resolved_doc"] or "unscoped")}'
            f'<div class="prism-num" style="text-align:left;margin-top:.35rem">'
            f'{run["stats"]["total_tokens"]:,} tok · '
            f'{run["stats"]["elapsed_ms"] / 1000:.1f}s</div></td>',
        ]
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (RESULTS_TABLE_CSS + '<div class="prism-results-shell">' +
            f'<table class="prism-results"><colgroup>{cols}</colgroup>'
            f"<thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"
            "</div>")


def convergence_donut(passed: int, failed: int):
    source = pd.DataFrame({"Result": ["Converged", "Diverged"],
                           "Questions": [passed, failed]})
    return (
        alt.Chart(source)
        .mark_arc(innerRadius=52, outerRadius=88, stroke="white", strokeWidth=2)
        .encode(
            theta=alt.Theta("Questions:Q", stack=True),
            color=alt.Color(
                "Result:N",
                scale=alt.Scale(domain=["Converged", "Diverged"],
                                range=[PASS_COLOR, FAIL_COLOR]),
                legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Result:N", "Questions:Q"],
        )
        .properties(height=230)
    )


def render_batch(runs: list, company: str, phase: str, model: str):
    section(f"{company} benchmark", f"{len(runs)} questions · {phase} · OpenAI {model}",
            ":material/analytics:")

    passed = sum(1 for r in runs if r["verdict"]["passed"] is True)
    failed = len(runs) - passed
    total_tokens = sum(r["stats"]["total_tokens"] for r in runs)
    total_seconds = sum(r["stats"]["elapsed_ms"] for r in runs) / 1000
    avg_seconds = total_seconds / len(runs) if runs else 0

    with st.container(border=True, key="benchmark-summary"):
        chart_col, metric_col = st.columns([1, 2], vertical_alignment="center")
        with chart_col:
            st.altair_chart(convergence_donut(passed, failed), width="stretch")
        with metric_col:
            top = st.columns(2, border=True)
            top[0].metric("Convergence", f"{100 * passed / len(runs):.0f}%" if runs else "—",
                          f"{passed} of {len(runs)} questions", icon=":material/task_alt:")
            top[1].metric("Diverged", failed, "awaiting review or governance",
                          delta_color="off", icon=":material/report:")
            bottom = st.columns(3, border=True)
            bottom[0].metric("Total tokens", f"{total_tokens:,}", icon=":material/token:")
            bottom[1].metric("Total time", f"{total_seconds:.1f}s", icon=":material/timer:")
            bottom[2].metric("Avg latency", f"{avg_seconds:.1f}s", icon=":material/speed:")

    st.html(results_table_html(runs))
    st.caption("Pass means convergence with the gold answer's chosen convention, not "
               "an assertion that other defensible conventions are objectively wrong.")

    # A failing row raises "why?", and the trace is the answer - so make it
    # reachable without re-running the question on its own.
    st.divider()
    section("Inspect a run", "Full pipeline trace for any question in this batch",
            ":material/manage_search:")
    options = {
        f"{'✅' if r['verdict']['passed'] else '❌'}  "
        f"{r['question']['id'].replace('financebench_id_', '#')} · "
        f"{r['question']['question'][:70]}": r
        for r in runs
    }
    chosen = st.selectbox("Question", list(options), label_visibility="collapsed")
    if chosen:
        render_detail(options[chosen])


def render_detail(run: dict):
    q, result, verdict, events, stats = (run["question"], run["result"], run["verdict"],
                                         run["events"], run["stats"])
    top = st.container(horizontal=True, vertical_alignment="center")
    with top:
        st.subheader(q["id"])
        status_badge(verdict["passed"])
        st.badge(result["resolved_doc"] or "Unscoped", icon=":material/description:",
                 color="blue")
    # Only present when Document resolution = AI (FTS shortlist + LLM pick) -
    # the deterministic path has no comparable "why" to show, it's regex.
    detail = result.get("resolution_detail")
    if detail:
        picked = detail.get("selected_documents") or []
        if picked:
            st.caption("Resolved by AI: " + "; ".join(
                f"{d['doc_name']} — {d.get('reason', '')}" for d in picked))
        else:
            st.caption("AI resolution declined — no candidate matched: "
                      + "; ".join(detail.get("missing_evidence") or ["no reason given"]))

    with st.container(border=True, key="answer-card"):
        st.markdown(f"### {q['question']}")
        cols = st.columns(2)
        with cols[0]:
            st.caption("Custom question — no reference" if q["id"] == "custom"
                      else "Gold answer")
            if q["id"] == "custom":
                st.write(q["expected_answer"])
            else:
                # concept/formula/evidence are ftsprism-native fields (see
                # eval/corpora/ftsprism.py's questions()) - absent for any
                # corpus that doesn't carry them, so each line only appears
                # when there's something real to show. Shown inline, not
                # tucked in an expander - there's room for it, and it's the
                # whole point of a gold answer: it should be checkable at a
                # glance, not one more click away. Deliberately NOT showing
                # keywords/tags here - those are retrieval-demo plumbing, not
                # part of what the gold answer actually asserts.
                gold_lines = []
                if q.get("concept"):
                    gold_lines.append(f"concept: {q['concept']}")
                if q.get("formula"):
                    gold_lines.append(f"formula: {q['formula']}")
                gold_lines.append(f'expected_answer: "{q["expected_answer"]}"')
                evidence = q.get("evidence") or []
                if evidence:
                    gold_lines.append("")
                    gold_lines.append("Evidence:")
                    for ev in evidence:
                        parts = [p for p in (
                            f"Page {ev['page']}" if ev.get("page") is not None else None,
                            ev.get("title"), ev.get("type")) if p]
                        gold_lines.append(f"{q.get('doc_name', '')}: " + " · ".join(parts))
                        if ev.get("text"):
                            gold_lines.append("")
                            gold_lines.append(ev["text"].strip())
                st.code("\n".join(gold_lines), language=None, wrap_lines=True)
        with cols[1]:
            st.caption("PRISM answer")
            # calc["computed"]'s formula is shown directly, not left to the
            # model's own prose - deterministic and immune to whatever
            # formatting inconsistency the model's text might have, same
            # reasoning as the plain-prose rule in answer.py.
            for c in result["calculation"].get("computed") or []:
                st.code(f"{c['label']}: {c['formula']} = {c['value']:.4g}",
                       language=None, wrap_lines=True)
            st.write(result["answer"])
        st.caption(verdict.get("comment", ""))

    metrics = st.columns(4, border=True)
    metrics[0].metric("Elapsed", f"{stats['elapsed_ms'] / 1000:.2f}s",
                      icon=":material/timer:")
    metrics[1].metric("Tokens", f"{stats['total_tokens']:,}", icon=":material/token:")
    metrics[2].metric("LLM calls", stats["llm_calls"], icon=":material/smart_toy:")
    metrics[3].metric("SQL++ calls", stats["sql_calls"], icon=":material/database:")

    tab_flow, tab_evidence = st.tabs([
        ":material/account_tree: How PRISM answered",
        ":material/find_in_page: Evidence",
    ])

    with tab_flow:
        render_pipeline_trace(run)

        calc = result["calculation"]
        if calc["computed"] and not calc["governed"]:
            with st.container(border=True):
                st.markdown("#### :material/how_to_reg: Governed correction")
                st.caption("Choose the convention this organization considers authoritative. "
                           "The next run executes it deterministically.")
                options = {f"{c['label']} · {c['value']:.6g}": c
                           for c in calc["computed"]}  # label is method_name
                CUSTOM = "Write my own convention…"
                picked = st.selectbox("Approved convention", list(options) + [CUSTOM],
                                      key=f"approve_{q['id']}")

                # The human must be able to approve a convention the model did
                # not propose. Restricting approval to the model's menu makes the
                # loop ratification, not governance - and it bites: the planner
                # stopped naming the facts one standard convention needs, so that
                # convention stopped being proposable, and it could no longer be
                # approved here at all.
                formula = None
                if picked == CUSTOM:
                    bound = sorted({f.get("name") for f in result["bound_facts"]
                                    if f.get("name")})
                    default = options[list(options)[0]]["formula"] if options else ""
                    formula = st.text_input(
                        "Formula", value=default, key=f"formula_{q['id']}",
                        help="Identifiers, numbers, parentheses and + - * / only. "
                             "Each identifier must be a fact the binder can locate "
                             "in the source.")
                    problem = runtime.validate_candidate({
                        "formula": formula,
                        "required_facts": [{"id": i} for i in
                                           retrieval.formula_identifiers(formula)]})
                    used = sorted(retrieval.formula_identifiers(formula))
                    if problem and not used and formula.strip():
                        # validate_candidate reports "references no facts" for an
                        # unparseable formula, which reads as the wrong problem.
                        st.error("Could not parse this formula. Use identifiers, "
                                 "numbers, parentheses and + - * / only.",
                                 icon=":material/error:")
                    elif problem:
                        st.error(problem, icon=":material/error:")
                    elif used:
                        unseen = [u for u in used if u not in bound]
                        st.caption(f"References: {', '.join(used)}")
                        if unseen:
                            st.warning(
                                f"Not bound in this run: {', '.join(unseen)}. The next "
                                "run must locate them or the formula will not evaluate.",
                                icon=":material/info:")
                    formula = None if problem else formula
                else:
                    formula = options[picked]["formula"]

                # A judgment question cannot be answered by the formula alone -
                # "is 0.96 healthy?" is a threshold opinion, and PRISM declines
                # the verdict without an approved policy. Defaulting this off
                # let a judgment be approved half-governed: the next run
                # computed a number and still refused to characterise it, which
                # reads as a bug rather than as the intended discipline.
                needs_policy = result["answer_kind"] == "judgment"
                use_threshold = st.toggle(
                    "Also approve an interpretation threshold", value=needs_policy,
                    key=f"threshold_toggle_{q['id']}",
                    help="Required to answer a judgment question. The formula alone "
                         "produces a number; deciding whether that number is healthy "
                         "is a separate, separately-versioned decision.")
                threshold = st.number_input("Healthy at or above", value=1.0, step=0.1,
                                            disabled=not use_threshold,
                                            key=f"threshold_{q['id']}")
                if needs_policy and not use_threshold:
                    st.warning("This question asks for a judgment. Without a threshold "
                               "PRISM will compute the value and still decline the "
                               "verdict.", icon=":material/info:")

                if st.button("Approve and learn", type="primary", disabled=not formula,
                             icon=":material/verified:", key=f"approve_button_{q['id']}"):
                    dictionary.approve(result["concept"], formula,
                                       threshold if use_threshold else None)
                    st.session_state.pop("showcase_runs", None)
                    st.toast("Approved. Re-run the question to see governed behavior.",
                             icon=":material/check_circle:")
                    st.rerun()

    with tab_evidence:
        section("Retrieved evidence", f"{len(result['chunks'])} chunks, in answer order",
                ":material/find_in_page:")
        if any(c.get("channels") for c in result["chunks"]):
            st.caption("A chunk both channels rank highly outranks one only a single "
                       "channel found. Ranks are compared, never raw scores — that is "
                       "the whole point of fusing by rank. `Moved` compares against "
                       "the previous run of this question, so a dial change shows as "
                       "specific chunks changing place.")
            previous = st.session_state.get(f"ranks_{q['id']}") or {}
            rows, current = [], {}
            for i, c in enumerate(result["chunks"], 1):
                channels = c.get("channels") or {}
                key = c.get("id")
                current[key] = i
                was = previous.get(key)
                rows.append({
                    "Final": i, "Page": c.get("page"), "Type": c.get("type"),
                    "BM25 rank": (channels.get("bm25") or {}).get("rank", "n/a"),
                    "Vector rank": (channels.get("vector") or {}).get("rank", "n/a"),
                    "Fused score": round(c.get("rrf_score") or c.get("score") or 0, 6),
                    "Moved": ("new" if was is None else
                              "—" if was == i else f"{'↑' if was > i else '↓'}{abs(was - i)}"),
                })
            st.dataframe(rows, hide_index=True, width="stretch")
            dropped = [k for k in previous if k not in current]
            if previous and (dropped or any(r["Moved"] not in ("—", "new") for r in rows)):
                st.caption(f"{sum(1 for r in rows if r['Moved'] == 'new')} entered, "
                           f"{len(dropped)} dropped out since the previous run.")
            st.session_state[f"ranks_{q['id']}"] = current
        elif any(c.get("score") is not None for c in result["chunks"]):
            # Native fusion returns one blended number and no derivation:
            # SEARCH_META exposes only id and score, and "explain": true adds
            # nothing through N1QL. Showing the score without pretending it can
            # be decomposed is the honest version - and the contrast with the
            # RRF table above is the reason the switch is worth having.
            st.caption("Couchbase native fusion · the Search service sums the lexical "
                       "and vector contributions into one SEARCH_SCORE(). The split "
                       "between them is not recoverable through N1QL.")
            st.dataframe([{
                "Final": i, "Page": c.get("page"), "Type": c.get("type"),
                "SEARCH_SCORE()": round(c["score"], 6) if c.get("score") is not None else None,
            } for i, c in enumerate(result["chunks"], 1)],
                hide_index=True, width="stretch")

        for i, chunk in enumerate(result["chunks"], 1):
            channels = chunk.get("channels") or {}
            if channels:
                source = " + ".join(
                    f"{name} " + (f"rank {v['rank']} " if v.get("rank") else "")
                    + f"w={v.get('weight', 1)} → {v['contribution']}"
                    for name, v in sorted(channels.items()))
                source += (f" = {chunk.get('rrf_score') or chunk.get('score'):.6f}")
            elif chunk.get("score") is not None:
                source = f"SEARCH_SCORE() {chunk['score']:.4f}"
            else:
                score = chunk.get("anchor_score")
                source = (f"anchor score {score}" if score is not None
                          else "semantic retrieval")
            with st.expander(
                f"[{i}] Page {chunk.get('page')} · {chunk.get('type')} · {source}",
                icon=":material/table_view:" if chunk.get("type") == "table"
                else ":material/article:",
            ):
                if channels:
                    st.json(channels, expanded=True)
                st.caption(f"Associated titles: {chunk.get('titles')}")
                st.code(chunk.get("text") or "", wrap_lines=True)

def fts_sidebar():
    st.caption("Search Vector Index")
    definition_path = REPO_ROOT / "design" / "fts-index.json"
    if definition_path.exists():
        with st.expander(config.FTS_DOCS_INDEX, icon=":material/manage_search:"):
            st.code(definition_path.read_text(), language="json", wrap_lines=True)
    else:
        st.code(
            f"{config.FTS_DOCS_INDEX}\n"
            "text-to-embed       BM25\n"
            "text-embedding      vector (2048d)\n"
            "associated-titles   text boost",
            language=None,
            wrap_lines=True,
        )
        st.caption("Add `design/fts-index.json` to show the complete definition here.")


# --------------------------------------------------------------------- page

questions = load_questions()
catalog_docs = load_catalog()
dictionary_data = dictionary.load()
catalog_names = {d.get("doc_name") for d in catalog_docs}
companies = sorted({q["company"] for q in questions if q.get("doc_name") in catalog_names})
if not companies:
    companies = sorted({q["company"] for q in questions})

with st.sidebar:
    brand = st.container(horizontal=True, vertical_alignment="center", gap="small")
    with brand:
        st.image(PRISM_MARK, width=68)
        with st.container(gap=None):
            st.markdown("### Couchbase Prism")
            st.caption("Governed retrieval that works immediately—and learns from reviewed use")

    st.space("small")
    st.markdown("### Run configuration")
    company = st.selectbox("Company", companies, index=companies.index("3M") if "3M" in companies else 0)
    company_questions = [q for q in questions if q["company"] == company]
    labels = ["All questions"] + [f"{q['id']} · {q['question'][:56]}" for q in company_questions]
    selection = st.selectbox("Question", labels)
    custom_question = st.text_area(
        "Or ask your own question", placeholder=f"Ask anything about {company}'s filings…",
        help="Runs the same pipeline, scoped to the company selected above. There is "
             "no gold reference for a custom question, so it is not scored — "
             "the trace and answer still show in full. Overrides the selection above "
             "when non-empty. Wrap a span in #hashes# (e.g. `#John Doe#`) to force an "
             "exact-phrase match on it alongside the usual BM25 leg — good for a named "
             "individual or exact term BM25's bag-of-words alone won't disambiguate.")
    phase = st.selectbox("Retrieval architecture", sorted(phases.PHASES),
                         index=sorted(phases.PHASES).index(phases.DEFAULT_PHASE),
                         format_func=lambda p: phases.LABELS.get(p, p))
    st.caption(PHASE_HELP[phase])

    if phases.get(phase).bm25:
        fusion = st.segmented_control(
            "Hybrid fusion", list(FUSION_LABELS), default="score", required=True,
            width="stretch", format_func=lambda f: FUSION_LABELS[f])
        st.caption(FUSION_HELP[fusion])
        tuning = tuning_panel(fusion)
    else:
        fusion = None
        tuning = {}

    provider = st.segmented_control("Model provider", ["OpenAI", "Amazon Bedrock"],
                                    default="OpenAI", required=True, width="stretch")
    if provider == "OpenAI":
        picked = {}
        for role, label, default, help_text in MODEL_ROLES:
            picked[role] = st.selectbox(
                label, MODEL_OPTIONS,
                index=MODEL_OPTIONS.index(default) if default in MODEL_OPTIONS else 0,
                accept_new_options=True, help=help_text, key=f"model_{role}")
        model = config.stage_models(**picked)
    else:
        st.info("Bedrock adapter is the next provider integration.",
                icon=":material/upcoming:")
        model = config.OPENAI_MODEL

    st.space("small")
    st.markdown("### Governance")
    approved = [e for e in dictionary_data.get("entries", [])
                if e.get("governance", {}).get("status") == "approved"]
    st.metric("Approved entries", len(approved), border=True)
    with st.expander("Dictionary entries", icon=":material/menu_book:"):
        if approved:
            # An id alone doesn't say what was approved, and the two entry types
            # are the point: a metric without a policy computes a value but
            # cannot deliver a verdict.
            metrics = [e for e in approved if e.get("entry_type") == "metric"]
            policies = {e.get("applies_to"): e for e in approved
                        if e.get("entry_type") == "interpretation_policy"}
            for entry in metrics:
                st.markdown(f"**{entry.get('id')}**")
                st.code(entry.get("interpretation", {}).get("formula", ""), language=None)
                policy = policies.get(entry.get("id"))
                if policy:
                    rules = ", ".join(f"{k} {v}" for k, v in
                                      (policy.get("policy") or {}).items())
                    st.caption(f":material/gavel: policy · {rules}")
                else:
                    st.caption(":material/warning: no interpretation policy — computes a "
                               "value, declines any verdict")
            orphans = [p for a, p in policies.items()
                       if a not in {e.get("id") for e in metrics}]
            for policy in orphans:
                st.caption(f"policy with no metric: {policy.get('id')}")
            # Resetting is a demo operation, not an accident to guard against:
            # the cold half of the two-pass story needs an empty dictionary, and
            # dropping to a terminal mid-demo breaks the narrative.
            if st.button("Reset dictionary", icon=":material/restart_alt:",
                         width="stretch",
                         help="Empty the dictionary to demonstrate the ungoverned "
                              "path again. Equivalent to `manage.py reset-dictionary`."):
                dictionary.clear()
                st.session_state.pop("showcase_runs", None)
                st.toast("Dictionary cleared — concepts are ungoverned again.",
                         icon=":material/restart_alt:")
                st.rerun()
        else:
            st.caption("Empty. PRISM still operates; entries arrive from reviewed use.")

if custom_question.strip():
    # No gold reference exists for a question nobody wrote a gold
    # answer for - expected_answer is a display string, not data judge.score
    # runs against; run_one() checks question["id"] == "custom" and skips
    # scoring entirely rather than judging against this placeholder text.
    selected_question = {
        "id": "custom", "question": custom_question.strip(), "doc_name": None,
        "expected_answer": "— (custom question, no benchmark reference)",
        "company": company,
    }
else:
    selected_question = None if selection == "All questions" else company_questions[
        labels.index(selection) - 1
    ]

# Switching the company, the question, or editing the custom-question box
# should clear whatever an EARLIER selection produced - a stale answer sitting
# under a newly-chosen question reads as if it belongs to it. This runs before
# the prompt-card so a changed selection shows only the question + Run button
# on the very same rerun the widget change already triggers; it does not fire
# on the rerun a button click itself causes, since the identity below is
# unchanged from the run immediately before that click.
selection_key = (company, selection, custom_question.strip())
if st.session_state.get("last_selection_key") != selection_key:
    st.session_state.pop("showcase_runs", None)
    st.session_state["last_selection_key"] = selection_key

tab_setup, tab_ask, tab_workbench = st.tabs(
    [":material/settings: Setup", ":material/chat: Ask a question",
     ":material/find_in_page: Document workbench"])

with tab_setup:
    st.caption("Everything a fresh environment needs before the first question can "
               "be asked - stage sample PDFs, point a Couchbase AI Data Plane "
               "workflow at them, then build the search index, catalog and "
               "dictionary from what it ingests. Each section runs independently - "
               "coming back to run Initialize alone, without re-uploading, is the "
               "normal case, not a special one.")

    section("Upload sample PDFs to S3", icon=":material/upload:",
            caption="Straight upload, no S3 object metadata, no per-company "
                    "subfolder - flat into the one folder below, matching what "
                    "the ingestion workflow and config.source_filename() both "
                    "expect. Bucket/folder/region come from .env, the same "
                    "values retrieval uses - not re-typed here, so they can't "
                    "drift out of sync with each other. Credentials are used "
                    "for this upload only - never written to disk, a config "
                    "file, or session state beyond the click that submits them.")
    pdf_root = REPO_ROOT / "eval" / "corpora" / "ftsprism" / "pdfs"
    available = s3_upload.find_pdfs(pdf_root)
    by_company = {}
    for company, path in available:
        by_company.setdefault(company, []).append(path)
    s3_bucket_env = os.environ.get("AWS_BUCKET")
    s3_folder_env = os.environ.get("AWS_FOLDER")
    if not available:
        st.info(f"No PDFs found under {pdf_root} yet.", icon=":material/info:")
    elif not (s3_bucket_env and s3_folder_env):
        st.error("AWS_BUCKET and AWS_FOLDER must be set in .env before "
                 "uploading - see .env.example.", icon=":material/error:")
    else:
        st.caption(", ".join(f"{c} ({len(p)})" for c, p in sorted(by_company.items()))
                  + f" — {len(available)} PDF(s) total, uploading flat to "
                    f"s3://{s3_bucket_env}/{s3_folder_env.strip('/')}/")
        with st.form("s3_upload_form"):
            s3_access_key = st.text_input("AWS access key ID", type="password")
            s3_secret_key = st.text_input("AWS secret access key", type="password")
            s3_session_token = st.text_input(
                "AWS session token (optional, for temporary credentials)",
                type="password")
            submitted = st.form_submit_button(
                "Upload to S3", icon=":material/upload:", width="stretch")
        if submitted:
            if not (s3_access_key and s3_secret_key):
                st.error("Access key and secret key are required.")
            else:
                progress_bar = st.progress(0.0)
                progress_line = st.empty()

                def on_s3_progress(i, total, company, filename, ok, error=None):
                    progress_bar.progress(i / total)
                    progress_line.caption(
                        f"[{i}/{total}] {filename}"
                        + ("" if ok else f" — ERROR: {error}"))

                result = s3_upload.upload_pdfs(
                    pdf_root, access_key=s3_access_key, secret_key=s3_secret_key,
                    session_token=s3_session_token.strip() or None,
                    on_progress=on_s3_progress)
                if result["failed"]:
                    st.warning(f"{len(result['uploaded'])}/{len(available)} uploaded, "
                              f"{len(result['failed'])} failed.")
                    st.error("\n".join(f"{f['key']}: {f['error']}"
                                      for f in result["failed"]))
                else:
                    st.success(f"{len(result['uploaded'])}/{len(available)} PDF(s) "
                              f"uploaded to s3://{s3_bucket_env}/{s3_folder_env.strip('/')}/")

    st.space("medium")
    section("Create the ingestion workflow", icon=":material/account_tree:",
            caption="Brief, on purpose - creating this from PRISM itself is a "
                    "stretch goal, not built yet.")
    st.markdown(
        "1. In Capella, open **AI Services → Workflows** and create a new workflow.\n"
        f"2. Point its source at the S3 bucket/folder used above.\n"
        f"3. Set the destination to `{config.BUCKET}.{config.SCOPE}.{config.DOCS_COLLECTION}` "
        "— bucket, scope and collection names are fixed (see the Search Vector "
        "Index panel below for what's already mapped there) and must match exactly.\n"
        "4. Run the workflow - it chunks, embeds, and stores every PDF as chunks "
        "in that collection.\n"
        "5. Once it finishes, run **Initialize** below.")

    st.space("medium")
    section("Initialize", icon=":material/bolt:",
            caption=f"{len(catalog_docs)} catalogued document(s). Run this after "
                    "the Couchbase AI Data Plane workflow finishes ingesting - no "
                    "manual index setup needed. Independent of the upload above - "
                    "coming back to run this alone, on whatever's already been "
                    "ingested, is the normal case.")
    with st.expander("Initialize environment", icon=":material/bolt:"):
        st.caption("Rebuilds the **search index** from `design/fts-index.json` "
                   "first and waits for it to fully catch up (well under a "
                   "minute for this corpus), then rebuilds the **catalog** "
                   "from ingested chunks through that index (no PDF access) "
                   "and empties the **dictionary**. Search index first because "
                   "the catalog rebuild reads through it too - both go through "
                   "the same index, nothing else. **`docs` and the ingestion "
                   "workflow are never touched.** Equivalent to "
                   "`manage.py initialize`.")
        if st.button("Initialize", icon=":material/bolt:", width="stretch",
                     help="Destructive: empties the catalog and dictionary and "
                          "rebuilds the search index. docs/chunks are untouched."):
            with st.status("Initializing…", expanded=True) as status:
                try:
                    summary = run_initialize(status, all_sectors())
                except Exception as exc:
                    status.update(label=f"Initialize failed: {exc}", state="error",
                                  expanded=True)
                    st.session_state["initialize_failures"] = []
                    st.exception(exc)
                    st.stop()
                ok = sum(1 for r in summary["catalog_results"] if r["ok"])
                failed = [r for r in summary["catalog_results"] if not r["ok"]]
                status.update(
                    label=f"Initialized — catalog {ok}/{len(summary['catalog_results'])}, "
                          f"dictionary cleared ({len(summary['dictionary_removed'])} "
                          "entrie(s)), search index rebuilt",
                    state="complete", expanded=False)
            # st.toast survives the st.rerun() below; st.success/st.error here
            # would not - same reasoning as the earlier catalog-only button.
            st.toast(f"Initialized — catalog {ok}/{len(summary['catalog_results'])}, "
                    "dictionary and search index rebuilt.",
                    icon=":material/check_circle:" if not failed else ":material/warning:")
            st.session_state["initialize_failures"] = failed
            load_catalog.clear()
            st.rerun()
        failures = st.session_state.get("initialize_failures")
        if failures:
            st.error("\n".join(f"{r['doc_name']}: {r['error']}" for r in failures))
    st.space("medium")
    fts_sidebar()

with tab_ask:
    with st.container(border=True, key="prompt-card"):
        prompt_area, action_area = st.columns([8, 1.25], vertical_alignment="center",
                                              gap="medium")
        with prompt_area:
            st.caption("Selected question" if selected_question else "Benchmark run")
            if selected_question:
                st.markdown(f"#### {selected_question['question']}")
                if selected_question["id"] == "custom":
                    st.caption("Custom question · not scored — doc resolution runs "
                              "normally, there is just no reference to grade against")
                else:
                    st.caption(f"{selected_question['id']} · expected document: "
                              f"{selected_question['doc_name']}")
            else:
                st.markdown(f"#### Run all {len(company_questions)} {company} questions")
                st.caption("The results dashboard will compare the gold answer and PRISM's answer.")
        with action_area:
            run_label = "Run all" if selected_question is None else "Run trace"
            run_clicked = st.button(run_label, type="primary", icon=":material/play_arrow:",
                                    width="stretch", disabled=provider != "OpenAI")

    if run_clicked:
        targets = company_questions if selected_question is None else [selected_question]
        runs = []
        with st.status(f"Running {len(targets)} question{'s' if len(targets) != 1 else ''}…",
                       expanded=True) as status:
            for index, question in enumerate(targets, 1):
                status.write(f"{index}/{len(targets)} · {question['id']}")
                try:
                    runs.append(run_one(question, catalog_docs, dictionary_data, phase,
                                        model, fusion, tuning))
                except Exception as exc:
                    status.update(label=f"Run stopped: {exc}", state="error", expanded=True)
                    st.exception(exc)
                    st.stop()
            status.update(label="Run complete", state="complete", expanded=False)
        st.session_state["showcase_runs"] = runs
        st.session_state["showcase_mode"] = "batch" if selected_question is None else "detail"
        st.session_state["showcase_context"] = {"company": company, "phase": phase, "model": model}

    runs = st.session_state.get("showcase_runs")
    if runs:
        context = st.session_state.get("showcase_context", {})
        if st.session_state.get("showcase_mode") == "batch":
            render_batch(runs, context.get("company", company), context.get("phase", phase),
                         context.get("model", model))
        else:
            render_detail(runs[0])
    else:
        with st.container(border=True, horizontal_alignment="center"):
            st.markdown("### Choose a company and question to begin")
            st.caption("Run all questions for the evaluation dashboard, or select one for a full trace.")

WORKBENCH_FETCH_LIMIT = 50

with tab_workbench:
    st.caption("Look up the raw ingested chunks for a company's document - reads "
               "straight from the docs collection through the Search Vector Index "
               "(BM25, same as the retrieval path), no PDF access needed. The "
               "embedding vector is left out of what's shown here; 2048 floats add "
               "nothing to read. Search terms search the WHOLE document, ignoring "
               "the page number - without terms, Page scopes to one page. Wrap a "
               "span in #hashes# (e.g. `#John Doe#`) to force an exact-phrase match "
               "on it alongside the usual BM25 terms.")
    wb_cols = st.columns([2, 3, 1, 2.5])
    with wb_cols[0]:
        wb_company = st.selectbox("Company", companies, key="wb_company")
    wb_doc_names = sorted({d.get("doc_name") for d in company_catalog(catalog_docs, wb_company)
                          if d.get("doc_name")})
    with wb_cols[1]:
        wb_doc = (st.selectbox("Document", wb_doc_names, key="wb_doc") if wb_doc_names
                  else st.selectbox("Document", ["(none catalogued)"], disabled=True))
    with wb_cols[2]:
        wb_page = st.number_input("Page", min_value=1, step=1, key="wb_page")
    with wb_cols[3]:
        wb_terms = st.text_input("Search terms (optional)", key="wb_terms",
                                 placeholder='e.g. total current assets, or #John Doe#')
    if st.button("Fetch", icon=":material/search:", key="wb_fetch",
                disabled=not wb_doc_names):
        terms = wb_terms.strip() or None
        wb_entry = next((d for d in company_catalog(catalog_docs, wb_company)
                         if d.get("doc_name") == wb_doc), None)
        chunks = fetch_chunks(wb_doc, page=None if terms else int(wb_page),
                              search_terms=terms, limit=WORKBENCH_FETCH_LIMIT,
                              source_filename=(wb_entry or {}).get("source_filename"))
        if not chunks:
            st.info("No chunks found for that document and filter.")
        else:
            note = (f"{len(chunks)} chunk(s) across the whole document" if terms
                    else f"{len(chunks)} chunk(s)")
            if len(chunks) == WORKBENCH_FETCH_LIMIT:
                note += (f" — showing the first {WORKBENCH_FETCH_LIMIT}; "
                        "narrow the search terms for more precision")
            st.caption(note)
            for chunk in chunks:
                meta = chunk.get("meta-data") or {}
                label = (f"p{meta.get('page-number', '?')} · {meta.get('type', '?')} · "
                        f"{chunk.get('element-id', '?')}")
                if terms:
                    # Score only means something as a relevance ranking - a
                    # plain page browse has no query to be relevant TO, so its
                    # label stays as-is rather than showing a misleadingly
                    # precise-looking number for a deterministic filter match.
                    label += f" · score {chunk.get('_score', 0):.3f}"
                with st.expander(label):
                    if terms:
                        # Strip #...# hashes before highlighting - the raw
                        # terms would search for the literal substring
                        # "#John" (never present in the text), silently
                        # highlighting nothing for exactly the words a phrase
                        # search cares most about.
                        highlight_words, _ = retrieval.extract_phrase_terms(terms)
                        st.markdown("**Matched text, terms highlighted:**")
                        st.html(highlight_terms(chunk.get("text-to-embed"), highlight_words))
                        st.divider()
                    st.json(chunk, expanded=True)
