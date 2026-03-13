import json
import re
from pathlib import Path
from urllib.parse import quote

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

ROOT = Path(__file__).resolve().parents[1]
VECTOR_INFO = ROOT / "ingestion" / "vector_store_info.json"

with open(VECTOR_INFO, encoding="utf-8") as f:
    vector_info = json.load(f)

vector_store_id = vector_info["vector_store_id"]

REPO_RESPONSE_SCHEMA = {
    "name": "repo_helper_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "supported_by_repo": {"type": "boolean"},
            "implementation_status": {"type": "string"},
            "answer": {"type": "string"},
            "repo_quotes": {
                "type": "array",
                "items": {"type": "string"}
            },
            "repo_paths": {
                "type": "array",
                "items": {"type": "string"}
            },
            "site_links": {
                "type": "array",
                "items": {"type": "string"}
            },
            "github_links": {
                "type": "array",
                "items": {"type": "string"}
            },
            "ai_generated_suggestion": {"type": "string"},
            "warning": {"type": "string"},
            "notes": {"type": "string"}
        },
        "required": [
            "supported_by_repo",
            "implementation_status",
            "answer",
            "repo_quotes",
            "repo_paths",
            "site_links",
            "github_links",
            "ai_generated_suggestion",
            "warning",
            "notes"
        ],
        "additionalProperties": False
    }
}


def ask_repo(question: str) -> dict:
    instructions = """
You are a developer assistant for the sde-sw-framework repository.

Use only retrieved repository content unless the repository does not fully specify the answer.

The retrieved documents contain metadata headers such as:
REPO_PATH:
SITE_URL:
GITHUB_URL:
FILE_NAME:

Rules:
1. Prefer repo-grounded answers.
2. Quote relevant repo text/snippets verbatim where possible.
3. repo_paths must list the REPO_PATH values from the retrieved content.
4. site_links must list SITE_URL values from the retrieved content where present.
5. github_links must list GITHUB_URL values from the retrieved content where present.
6. implementation_status must be one of:
   - fully_specified_in_repo
   - partially_specified_in_repo
   - not_specified_in_repo
7. If the repo does not fully specify the answer, still answer helpfully, but:
   - clearly say what is supported by the repo
   - put any extra proposed code or approach in ai_generated_suggestion
   - include a clear warning
8. If the repo does not support the answer at all, set supported_by_repo=false.
9. Assume the user is a technical analyst unless the question says otherwise.
10. Focus on implementation questions about deriving core dataset tables, variables, codelists, and processing logic.
11. Do not invent repo paths or URLs. Only return paths/links that are explicitly present in retrieved content.
12. If implementation_status is fully_specified_in_repo, ai_generated_suggestion should usually be an empty string unless the user explicitly asked for example code.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=instructions,
        input=question,
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id]
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": REPO_RESPONSE_SCHEMA["name"],
                "strict": REPO_RESPONSE_SCHEMA["strict"],
                "schema": REPO_RESPONSE_SCHEMA["schema"]
            }
        }
    )

    return json.loads(response.output_text)


def repo_path_to_github_url(repo_path: str) -> str | None:
    repo_path = (repo_path or "").strip()
    if not repo_path:
        return None

    github_repo_url = "https://github.com/Exeter-Diabetes/sde-sw-framework"
    github_branch = "dev"
    encoded_path = quote(repo_path.replace("\\", "/"), safe="/")
    return f"{github_repo_url}/blob/{github_branch}/{encoded_path}"


def display_site_links(links: list[str]) -> None:
    st.subheader("Site links")
    if links:
        seen = set()
        for link in links:
            if link and link not in seen:
                seen.add(link)
                st.markdown(f"- [Open documentation page]({link})")
    else:
        st.write("None")


def display_repo_paths(paths: list[str]) -> None:
    st.subheader("Repo paths")
    if paths:
        seen = set()
        for path in paths:
            if not path or path in seen:
                continue
            seen.add(path)
            github_url = repo_path_to_github_url(path)
            if github_url:
                st.markdown(f"- [{path}]({github_url})")
            else:
                st.write(f"- {path}")
    else:
        st.write("None")


def clean_quote_text(text: str) -> str:
    if not text:
        return ""

    cleaned = str(text)

    # Turn escaped newlines/tabs into spaces
    cleaned = cleaned.replace("\\n", " ")
    cleaned = cleaned.replace("\\t", " ")
    cleaned = cleaned.replace("\n", " ")
    cleaned = cleaned.replace("\t", " ")

    # Remove markdown links but keep visible text: [text](url) -> text
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)

    # Remove markdown heading markers
    cleaned = re.sub(r"(?<!\w)#{1,6}\s*", "", cleaned)

    # Remove backticks
    cleaned = cleaned.replace("`", "")

    # Remove emphasis markers
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("*", "")

    # Collapse repeated whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


st.set_page_config(
    page_title="SDE AI Helper",
    page_icon="🧠",
    layout="wide"
)

st.markdown(
    """
    <style>
    .warning-box {
        background-color: #fff7ed;
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #fdba74;
        margin-top: 1rem;
        margin-bottom: 1rem;
        color: #9a3412;
    }
    .quote-box {
        padding: 0.75rem 0.9rem;
        border-left: 4px solid #d1d5db;
        background-color: rgba(127, 127, 127, 0.08);
        margin-bottom: 0.6rem;
        border-radius: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("SDE AI Helper")
st.caption(
    "Repo-grounded assistant for implementation questions about the core diabetes dataset and related framework content."
)

with st.sidebar:
    st.subheader("Scope")
    st.markdown(
        """
- Answers are grounded in the local clone of `sde-sw-framework`
- Quotes and links come from indexed repo content
- If the repo does not fully specify an answer, the app can suggest an approach and will label it clearly
        """
    )
    st.subheader("Current vector store")
    st.code(vector_store_id)

question = st.text_area(
    "Ask an implementation question",
    height=140,
    placeholder="Example: Write a SQL skeleton to derive the earliest clean diabetes code per patient"
)

ask_button = st.button("Ask the repo helper")

if ask_button:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching repository content and generating grounded answer..."):
            try:
                result = ask_repo(question.strip())

                if result["supported_by_repo"]:
                    st.success("Supported by repository content.")
                else:
                    st.warning("The repository does not directly support this answer.")

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("Answer")
                    st.write(result["answer"])

                    st.subheader("Quoted repo text")
                    if result["repo_quotes"]:
                        for quote_text in result["repo_quotes"]:
                            cleaned_quote = clean_quote_text(quote_text)
                            if cleaned_quote:
                                st.markdown(
                                    f'<div class="quote-box">{cleaned_quote}</div>',
                                    unsafe_allow_html=True
                                )
                    else:
                        st.write("No repo quotes returned.")

                    if result["ai_generated_suggestion"]:
                        st.subheader("AI-generated suggestion")
                        st.markdown(
                            """
                            <div class="warning-box">
                            This section goes beyond explicit repo content and should be reviewed carefully.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.code(result["ai_generated_suggestion"], language="sql")

                with col2:
                    st.subheader("Support summary")
                    st.write(f"**Supported by repo:** {result['supported_by_repo']}")
                    st.write(f"**Implementation status:** {result['implementation_status']}")

                    display_site_links(result["site_links"])
                    display_repo_paths(result["repo_paths"])

                if result["warning"]:
                    st.subheader("Warning")
                    st.markdown(
                        f'<div class="warning-box">{result["warning"]}</div>',
                        unsafe_allow_html=True
                    )

                st.subheader("Notes")
                if result["notes"]:
                    st.write(result["notes"])
                else:
                    st.write("None")

                with st.expander("Raw structured response"):
                    st.json(result)

            except Exception as e:
                st.error("API call failed.")
                st.code(str(e))