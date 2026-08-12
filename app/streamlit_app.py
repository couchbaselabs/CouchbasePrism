"""PRISM showcase — the measured runtime, made inspectable.

Run from the repository root:

    streamlit run app/streamlit_app.py

The app deliberately calls ``prism.runtime.answer_question`` and
``eval.judge.score``. It does not maintain a friendlier demo-only pipeline.
"""
import html
import json
import pathlib
import sys
import time

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from eval import judge, phases  # noqa: E402
from eval.corpora import load as load_corpus  # noqa: E402
from prism import catalog, config, dictionary, runtime, trace  # noqa: E402


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

    /* REDUCE MAIN PANEL TOP SPACING (Adjust 1.5rem as needed) */
    padding-top: 1.5rem !important;

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
MODEL_OPTIONS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
PHASE_HELP = {
    "1-vector": "Baseline · pure vector kNN across the corpus (2/8)",
    "2-catalog": "Adds deterministic document scoping (3/8)",
    "3-hybrid": "Adds whole-question BM25 + kNN; no gain in this sample (3/8)",
    "4-planner": "Uses planner-generated content anchors (4/8)",
    "5-runtime": "Adds binding, calculation and governed semantics (4/8 cold, 5/8 learned)",
}
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PRISM_MARK = REPO_ROOT / "app" / "assets" / "prism-mark.png"


@st.cache_data(ttl=60, show_spinner=False)
def load_catalog():
    return catalog.load_all()


@st.cache_data(show_spinner=False)
def load_questions():
    return load_corpus("financebench").questions()


def company_catalog(catalog_docs: list, company: str) -> list:
    """Temporary eval adapter until catalog resolution is entity-aware."""
    prefix = company.upper().replace(" ", "") + "_"
    matched = [d for d in catalog_docs
               if (d.get("doc_name") or "").upper().replace(" ", "").startswith(prefix)]
    return matched or catalog_docs


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
            phase: str, model: str) -> dict:
    started = time.perf_counter()
    with trace.capture() as events:
        result = runtime.answer_question(
            question["question"], company_catalog(catalog_docs, question["company"]),
            dictionary_data, options=phases.get(phase), model=model)
        verdict = judge.score(question["question"], question["expected_answer"],
                              result["answer"], model=model)
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
    messages = (event.get("request") or {}).get("messages") or []
    text = "\n".join(str(m.get("content", "")) for m in messages).lower()
    if "planning what evidence is needed" in text:
        return "planner"
    if "binding named financial facts" in text:
        return "binder"
    if "proposing how a named financial concept" in text:
        return "candidate"
    if "grading whether a candidate answer" in text:
        return "judge"
    return "answer"


def show_llm_exchange(event: dict, label: str):
    request = event.get("request") or {}
    usage = event.get("usage") or {}
    elapsed = event.get("elapsed_ms", 0)
    st.caption(f"{label} · {request.get('model', 'model')} · "
               f"{usage.get('total_tokens', '—')} tokens · {elapsed:,.0f} ms")
    request_tab, response_tab = st.tabs(["Request", "Response"])
    with request_tab:
        st.json(request, expanded=False)
    with response_tab:
        st.code(event.get("response") or event.get("error") or "", wrap_lines=True)


def show_sql(event: dict, label: str):
    elapsed = event.get("elapsed_ms", 0)
    st.caption(f"{label} · {event.get('row_count', 0)} rows · {elapsed:,.0f} ms")
    st.code(event.get("statement", ""), language="sql", wrap_lines=True)
    with st.popover("Parameters and query metrics", icon=":material/query_stats:"):
        st.markdown("**Parameters**")
        st.json(event.get("params") or {}, expanded=True)
        if event.get("metrics"):
            st.markdown("**Metrics**")
            st.json(event["metrics"], expanded=True)


def render_pipeline_trace(run: dict):
    q, result, events = run["question"], run["result"], run["events"]
    llms = {role: [e for e in events if e.get("type") == "llm_call"
                   and llm_role(e) == role]
            for role in ("planner", "candidate", "binder", "answer", "judge")}
    sql_events = [e for e in events if e.get("type") == "couchbase_query"]
    embed_events = [e for e in events if e.get("type") == "embedding_call"]
    plan = result["plan"]
    options = result["options"]
    retrieval_mode = (
        "whole-question BM25 + vector kNN"
        if options.bm25 else
        "content anchors + vector kNN"
        if options.anchors else
        "vector kNN"
    )

    with st.expander("1 · Filter — resolve the governed document scope", expanded=True,
                     icon=":material/filter_alt:"):
        st.markdown("**In** · user question + company-filtered catalog  ")
        if options.catalog_filter:
            st.markdown("**Work** · parse fiscal period and filing type; resolve against the "
                        "catalog before semantic search. The catalog is loaded once, then this "
                        "per-question resolution is deterministic application code.  ")
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
        if llms["planner"]:
            show_llm_exchange(llms["planner"][0], "Evidence planner")
        st.markdown("**Out**")
        st.json(plan, expanded=True)

    with st.expander("3 · Retrieve — anchors + semantic search inside the document",
                     expanded=True, icon=":material/manage_search:"):
        st.markdown("**In** · resolved document key + content anchors + question embedding  ")
        st.markdown(f"**Active strategy** · `{retrieval_mode}`  ")
        if options.anchors:
            st.markdown("**Work** · exact content-anchor matches identify structurally relevant "
                        "chunks; vector search fills the remaining evidence budget. Results are "
                        "deduplicated with anchor hits first.  ")
        elif options.bm25:
            st.markdown("**Work** · send the full question to the Search Vector Index, combining "
                        "BM25 lexical scores with vector similarity inside the resolved document.  ")
        else:
            st.markdown("**Work** · retrieve by vector similarity only, scoped to the resolved "
                        "document when catalog filtering is enabled.  ")
        for event in embed_events:
            st.caption(f"Embedding · {event.get('dimensions', '—')} dimensions · "
                       f"{event.get('elapsed_ms', 0):,.0f} ms")
            st.json(event.get("request") or {}, expanded=False)
        for number, event in enumerate(sql_events, 1):
            show_sql(event, f"SQL++ {number}")
        st.markdown(f"**Out** · {len(result['chunks'])} evidence chunks")

    with st.expander("4 · Govern — apply approved semantics or surface ambiguity",
                     expanded=True, icon=":material/policy:"):
        st.markdown("**In** · planner concept + scoped dictionary  ")
        if result["dictionary_entry"]:
            st.success(f"Approved metric: `{result['dictionary_entry']}`")
        else:
            st.warning("No approved metric matched. Candidate interpretations may be proposed.")
        st.write("Interpretation policy:", "Approved" if result["has_policy"] else "None")
        if llms["candidate"]:
            show_llm_exchange(llms["candidate"][0], "Candidate-formula proposal")
        if result["candidates"]:
            st.json(result["candidates"], expanded=True)

    with st.expander("5 · Bind — connect each fact to a row, column and source",
                     expanded=bool(result["bound_facts"]), icon=":material/link:"):
        st.markdown("**In** · required fact names + retrieved evidence  ")
        st.markdown("**Work** · bind each value to entity, units, period, printed row and "
                    "source page; reject values absent from the retrieved text or mixed periods.  ")
        if llms["binder"]:
            show_llm_exchange(llms["binder"][0], "Fact binder")
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
        if llms["answer"]:
            show_llm_exchange(llms["answer"][0], "Answer synthesis")
        st.markdown("**Out**")
        st.info(result["answer"], icon=":material/auto_awesome:")

    with st.expander("8 · Evaluate — compare with the benchmark convention",
                     icon=":material/fact_check:"):
        st.caption("Evaluation-only. This is not part of a production answer path.")
        if llms["judge"]:
            show_llm_exchange(llms["judge"][0], "FinanceBench judge")
        status_badge(run["verdict"]["passed"])
        st.write(run["verdict"].get("comment", ""))


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
    ("Status", "9%"), ("Question", "19%"), ("FinanceBench answer", "21%"),
    ("PRISM answer", "25%"), ("Why", "15%"), ("Document", "11%"),
]


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
            f'<td>{html.escape(q["question"])}</td>',
            f'<td>{html.escape(q["expected_answer"])}</td>',
            f'<td>{html.escape(result["answer"])}</td>',
            f'<td>{html.escape(verdict.get("comment") or "")}</td>',
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

    st.markdown(results_table_html(runs), unsafe_allow_html=True)
    st.caption("Pass means convergence with FinanceBench's chosen convention, not an "
               "assertion that other defensible conventions are objectively wrong.")

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

    with st.container(border=True, key="answer-card"):
        st.markdown(f"### {q['question']}")
        cols = st.columns(2)
        with cols[0]:
            st.caption("FinanceBench reference")
            st.write(q["expected_answer"])
        with cols[1]:
            st.caption("PRISM answer")
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
                           for c in calc["computed"]}
                picked = st.selectbox("Approved convention", list(options),
                                      key=f"approve_{q['id']}")
                use_threshold = st.toggle("Also approve an interpretation threshold", value=False,
                                          key=f"threshold_toggle_{q['id']}")
                threshold = st.number_input("Healthy at or above", value=1.0, step=0.1,
                                            disabled=not use_threshold,
                                            key=f"threshold_{q['id']}")
                if st.button("Approve and learn", type="primary",
                             icon=":material/verified:", key=f"approve_button_{q['id']}"):
                    dictionary.approve(result["concept"], options[picked]["formula"],
                                       threshold if use_threshold else None)
                    st.session_state.pop("showcase_runs", None)
                    st.toast("Approved. Re-run the question to see governed behavior.",
                             icon=":material/check_circle:")
                    st.rerun()

    with tab_evidence:
        section("Retrieved evidence", f"{len(result['chunks'])} chunks, in answer order",
                ":material/find_in_page:")
        for i, chunk in enumerate(result["chunks"], 1):
            score = chunk.get("anchor_score")
            source = f"anchor score {score}" if score is not None else "semantic retrieval"
            with st.expander(
                f"[{i}] Page {chunk.get('page')} · {chunk.get('type')} · {source}",
                icon=":material/table_view:" if chunk.get("type") == "table"
                else ":material/article:",
            ):
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

    st.markdown("### Run configuration")
    company = st.selectbox("Company", companies, index=companies.index("3M") if "3M" in companies else 0)
    company_questions = [q for q in questions if q["company"] == company]
    labels = ["All questions"] + [f"{q['id']} · {q['question'][:56]}" for q in company_questions]
    selection = st.selectbox("Question", labels)
    phase = st.selectbox("Runtime phase", sorted(phases.PHASES),
                         index=sorted(phases.PHASES).index(phases.DEFAULT_PHASE))
    st.caption(PHASE_HELP[phase])

    provider = st.segmented_control("Model provider", ["OpenAI", "Amazon Bedrock"],
                                    default="OpenAI", required=True, width="stretch")
    if provider == "OpenAI":
        default_model = config.OPENAI_MODEL if config.OPENAI_MODEL in MODEL_OPTIONS else MODEL_OPTIONS[0]
        model = st.selectbox("Model", MODEL_OPTIONS,
                             index=MODEL_OPTIONS.index(default_model), accept_new_options=True)
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
            for entry in approved:
                st.code(entry.get("id", ""), language=None)
        else:
            st.caption("Empty. PRISM still operates; entries arrive from reviewed use.")

    st.space("medium")
    fts_sidebar()

selected_question = None if selection == "All questions" else company_questions[
    labels.index(selection) - 1
]
with st.container(border=True, key="prompt-card"):
    prompt_area, action_area = st.columns([8, 1.25], vertical_alignment="center",
                                          gap="medium")
    with prompt_area:
        st.caption("Selected question" if selected_question else "Benchmark run")
        if selected_question:
            st.markdown(f"#### {selected_question['question']}")
            st.caption(f"{selected_question['id']} · expected document: "
                       f"{selected_question['doc_name']}")
        else:
            st.markdown(f"#### Run all {len(company_questions)} {company} questions")
            st.caption("The results dashboard will compare FinanceBench and PRISM answers.")
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
                runs.append(run_one(question, catalog_docs, dictionary_data, phase, model))
            except Exception as exc:
                status.update(label=f"Run stopped: {exc}", state="error", expanded=True)
                st.exception(exc)
                st.stop()
        status.update(label="Run complete", state="complete", expanded=False)
    st.session_state["showcase_runs"] = runs
    st.session_state["showcase_mode"] = "batch" if selection == "All questions" else "detail"
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
