"""Skills is always a local file, never Couchbase (see prism/skills.py and
docs/adr/0003-skills-a-domain-expert-owned-knowledge-layer.md) - so unlike
dictionary/concepts, there is no Couchbase branch to mock here, only the
file round-trip and the prompt-rendering shape."""
from prism import skills


def test_load_is_empty_when_the_file_does_not_exist_yet(tmp_path):
    assert skills.load(scope="secfilings", path=tmp_path / "skills.yaml") == []


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "skills.yaml"
    skills.save(["A 10-Q covers only its own quarter.",
                "A proxy statement is dated a year after the year it discloses."],
               scope="secfilings", path=path)
    assert skills.load(scope="secfilings", path=path) == [
        "A 10-Q covers only its own quarter.",
        "A proxy statement is dated a year after the year it discloses.",
    ]


def test_save_drops_blank_lines(tmp_path):
    # A textarea save (app/streamlit_app.py's Skills tab) round-trips
    # st.text_area.splitlines() straight through - blank lines from normal
    # editing (a trailing newline, a spacer between statements) must not
    # become empty "skills" that get appended into the prompt as noise.
    path = tmp_path / "skills.yaml"
    skills.save(["First skill.", "", "   ", "Second skill.  "],
               scope="secfilings", path=path)
    assert skills.load(scope="secfilings", path=path) == \
        ["First skill.", "Second skill."]


def test_save_preserves_other_scopes(tmp_path):
    path = tmp_path / "skills.yaml"
    skills.save(["secfilings-only skill."], scope="secfilings", path=path)
    skills.save(["fhir-only skill."], scope="fhir", path=path)
    assert skills.load(scope="secfilings", path=path) == ["secfilings-only skill."]
    assert skills.load(scope="fhir", path=path) == ["fhir-only skill."]


def test_render_is_empty_string_with_no_skills(tmp_path):
    # Not a header with nothing under it - an empty scope must not add any
    # visible text to resolve_and_plan's system prompt.
    assert skills.render(scope="secfilings", path=tmp_path / "skills.yaml") == ""


def test_render_lists_each_skill_as_its_own_bullet(tmp_path):
    path = tmp_path / "skills.yaml"
    skills.save(["First skill.", "Second skill."], scope="secfilings", path=path)
    rendered = skills.render(scope="secfilings", path=path)
    assert "SKILLS" in rendered
    assert "- First skill." in rendered
    assert "- Second skill." in rendered
