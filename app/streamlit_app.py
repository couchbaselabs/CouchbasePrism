"""PRISM demo.

    streamlit run app/streamlit_app.py

Shows the runtime as a sequence of stages rather than a chat box, because the
story is the pipeline and the human decision inside it, not the prose at the
end. The interesting moment is when a concept has no approved entry: PRISM
computes several labeled candidates, declines to pick, and offers the approval
that makes it deterministic from then on.

It calls prism.runtime.answer_question - the same function the benchmark
harness calls - so what is demonstrated is exactly what is measured.
"""
import sys
import pathlib

import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from eval.corpora import load as load_corpus  # noqa: E402
from prism import catalog, config, dictionary, runtime  # noqa: E402

st.set_page_config(page_title="PRISM", layout="wide")


@st.cache_data(ttl=60)
def _catalog_docs():
    return catalog.load_all()


def stage(title: str, caption: str = None):
    st.markdown(f"##### {title}")
    if caption:
        st.caption(caption)


st.title("PRISM")
st.caption("Governed retrieval over document corpora on Couchbase — "
           "works immediately, learns from governed corrections.")

with st.sidebar:
    st.subheader("Dictionary")
    data = dictionary.load()
    entries = data.get("entries", [])
    metrics = [e for e in entries if e.get("entry_type") == "metric"]
    policies = [e for e in entries if e.get("entry_type") == "interpretation_policy"]
    st.metric("approved metrics", len(metrics))
    st.metric("interpretation policies", len(policies))
    if entries:
        for e in metrics:
            st.code(f"{e['id']}\n{e['interpretation']['formula']}", language=None)
    else:
        st.info("Empty. PRISM answers fine like this — entries arrive from use.")
    if entries and st.button("Reset dictionary (demo)"):
        dictionary.save({"entries": []})
        st.rerun()

    st.divider()
    st.caption(f"{config.BUCKET}.{config.SCOPE}")

corpus = load_corpus("financebench")
questions = corpus.questions(company="3M")
labels = [f"{q['id']} — {q['question'][:70]}" for q in questions]
choice = st.selectbox("Question", range(len(questions)), format_func=lambda i: labels[i])
selected = questions[choice]
question_text = st.text_area("", selected["question"], height=80)

if st.button("Run", type="primary"):
    with st.spinner("catalog → plan → retrieve → bind → compute → validate"):
        result = runtime.answer_question(question_text, _catalog_docs())
    st.session_state["result"] = result
    st.session_state["expected"] = selected["expected_answer"]

result = st.session_state.get("result")
if result:
    left, right = st.columns([3, 2])

    with left:
        stage("1 · Catalog", "resolve the document before any semantic search")
        st.success(f"`{result['resolved_doc']}`")

        stage("2 · Evidence plan", "no dictionary entry required to produce this")
        plan = result["plan"]
        st.write(f"**concept** `{plan.get('concept')}` · "
                 f"**kind** `{plan.get('answer_kind')}`")
        st.write("**content anchors** — verbatim row labels, not section titles "
                 "(those are unreliable in this corpus)")
        st.code("\n".join(plan.get("content_anchors") or []), language=None)

        stage("3 · Retrieval", f"{len(result['chunks'])} chunks, anchors ranked by rarity")
        for i, c in enumerate(result["chunks"][:6], 1):
            score = c.get("anchor_score")
            tag = f"anchor {score}" if score is not None else "vector"
            with st.expander(f"[{i}] page {c.get('page')} · {c.get('type')} · {tag}"):
                st.text((c.get("text") or "")[:2000])

        if result["bound_facts"]:
            stage("4 · Fact binding", "each value tied to a row, column, page — and "
                                       "checked against the source text")
            st.dataframe(
                [{"fact": f.get("name"), "value": f.get("value"),
                  "row": f.get("row_label"), "column": f.get("period"),
                  "page": f.get("source_page"),
                  "grounded": "✓" if f.get("grounded") else "✗"}
                 for f in result["bound_facts"]],
                use_container_width=True, hide_index=True)

    with right:
        calc = result["calculation"]
        conclusion = result["conclusion"]

        if calc["computed"]:
            if calc["governed"]:
                stage("5 · Calculation — governed",
                      "approved formula, evaluated in code")
                st.success(f"**{calc['computed'][0]['value']:.4g}**  \n"
                           f"`{calc['computed'][0]['formula']}`")
            else:
                stage("5 · Calculation — ungoverned",
                      "labeled candidates; not claimed exhaustive")
                for c in calc["computed"]:
                    st.warning(f"**{c['label']} = {c['value']:.4g}**  \n"
                               f"`{c['formula']}`  \n{c.get('rationale') or ''}")

        if conclusion:
            stage("6 · Validation")
            status = conclusion.get("status")
            if status == "agreed":
                st.success(f"verdict authorised — {conclusion['explanation']}")
            elif status == "no_policy":
                st.error("**Verdict declined.** No approved interpretation policy "
                         "exists, so no healthy/unhealthy characterisation is "
                         "authorised. PRISM may compute a metric without holding "
                         "authority to judge it.")
            elif status == "conflicting":
                st.error("**Verdict blocked.** Candidates straddle the approved "
                         "threshold — blocking on review rather than picking one.")

        if calc["computed"] and not calc["governed"]:
            stage("7 · Governed correction", "the entire human step")
            options = {f"{c['label']} = {c['value']:.4g}": c for c in calc["computed"]}
            picked = st.radio("Which convention is authoritative here?", list(options))
            threshold = st.number_input("Healthy at or above (optional)", value=1.0,
                                        step=0.1, format="%.2f")
            use_threshold = st.checkbox("Also approve this threshold", value=True)
            if st.button("Approve", type="primary"):
                dictionary.approve(
                    result["concept"], options[picked]["formula"],
                    threshold if use_threshold else None)
                st.success("Approved. Re-run — it is deterministic from here.")
                st.rerun()

        stage("Answer")
        st.write(result["answer"])
        with st.expander("Expected answer (benchmark reference)"):
            st.write(st.session_state.get("expected", ""))
