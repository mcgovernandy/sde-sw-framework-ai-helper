import json
from pathlib import Path
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


def ask_repo(question: str):
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

    result = json.loads(response.output_text)

    print("\nSupported by repo:")
    print(result["supported_by_repo"])

    print("\nImplementation status:")
    print(result["implementation_status"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nRepo quotes:")
    if result["repo_quotes"]:
        for q in result["repo_quotes"]:
            print(f'- "{q}"')
    else:
        print("- None")

    print("\nRepo paths:")
    if result["repo_paths"]:
        for p in result["repo_paths"]:
            print(f"- {p}")
    else:
        print("- None")

    print("\nSite links:")
    if result["site_links"]:
        for link in result["site_links"]:
            print(f"- {link}")
    else:
        print("- None")

    print("\nGitHub links:")
    if result["github_links"]:
        for link in result["github_links"]:
            print(f"- {link}")
    else:
        print("- None")

    print("\nAI-generated suggestion:")
    print(result["ai_generated_suggestion"] or "None")

    print("\nWarning:")
    print(result["warning"] or "None")

    print("\nNotes:")
    print(result["notes"] or "None")


if __name__ == "__main__":
    question = input("Ask a repo implementation question: ").strip()
    if question:
        ask_repo(question)
    else:
        print("No question entered.")