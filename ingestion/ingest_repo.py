import json
import shutil
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = ROOT / "config" / "settings.json"
OUTPUT_JSON = ROOT / "ingestion" / "vector_store_info.json"
PREPARED_DIR = ROOT / "prepared_docs"

with open(SETTINGS_PATH, encoding="utf-8") as f:
    settings = json.load(f)

source_repo = (ROOT / settings["source_repo_path"]).resolve()
site_base_url = settings["site_base_url"].rstrip("/")
github_repo_url = settings["github_repo_url"].rstrip("/")
github_branch = settings["github_branch"]

INCLUDE_PATHS = [
    source_repo / "README.md",
    source_repo / "docs",
    source_repo / "code_lists",
    source_repo / "projects",
]

ALLOWED_SUFFIXES = {".md", ".txt", ".csv", ".sql", ".yaml", ".yml", ".json", ".R", ".r", ".py"}


def collect_files():
    files = []

    for path in INCLUDE_PATHS:
        if not path.exists():
            continue

        if path.is_file():
            if path.suffix in ALLOWED_SUFFIXES or path.name == "README.md":
                files.append(path)
            continue

        for child in path.rglob("*"):
            if child.is_file() and child.suffix in ALLOWED_SUFFIXES:
                files.append(child)

    return sorted(set(files))


def make_site_url(relative_path: Path) -> str:
    rel = str(relative_path).replace("\\", "/")

    # README maps to root
    if rel == "README.md":
        return f"{site_base_url}/"

    # docs/index.md also maps to root
    if rel == "docs/index.md":
        return f"{site_base_url}/"

    # Remove docs/ prefix if present
    if rel.startswith("docs/"):
        rel = rel[len("docs/"):]

    # Convert markdown paths to mkdocs directory URLs
    if rel.endswith(".md"):
        rel = rel[:-3]

    return f"{site_base_url}/{rel}/"


def make_github_url(relative_path: Path) -> str:
    rel = str(relative_path).replace("\\", "/")
    return f"{github_repo_url}/blob/{github_branch}/{rel}"


def prepare_file(source_file: Path) -> Path:
    relative_path = source_file.relative_to(source_repo)
    site_url = make_site_url(relative_path)
    github_url = make_github_url(relative_path)

    try:
        body = source_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        body = source_file.read_text(encoding="latin-1")

    prepared_text = f"""REPO_PATH: {relative_path.as_posix()}
SITE_URL: {site_url}
GITHUB_URL: {github_url}
FILE_NAME: {source_file.name}

CONTENT_START
{body}
"""

    safe_name = str(relative_path).replace("\\", "__").replace("/", "__")
    output_file = PREPARED_DIR / f"{safe_name}.txt"
    output_file.write_text(prepared_text, encoding="utf-8")

    return output_file


def main():
    files = collect_files()

    if not files:
        raise ValueError("No eligible repo files found to ingest.")

    if PREPARED_DIR.exists():
        shutil.rmtree(PREPARED_DIR)
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)

    prepared_files = []
    print("Preparing files:")
    for f in files:
        rel = f.relative_to(source_repo)
        print(f" - {rel}")
        prepared_files.append(prepare_file(f))

    vector_store = client.vector_stores.create(
        name="SDE SW Framework AI Helper"
    )

    print(f"\nCreated vector store: {vector_store.id}")

    streams = [open(f, "rb") for f in prepared_files]
    try:
        batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id,
            files=streams
        )
    finally:
        for s in streams:
            s.close()

    print("\nUpload complete.")
    print(f"Batch status: {batch.status}")
    print(f"File counts: {batch.file_counts}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "vector_store_id": vector_store.id,
                "name": vector_store.name,
                "source_repo_path": str(source_repo),
                "uploaded_files": [str(f.relative_to(source_repo)) for f in files]
            },
            f,
            indent=2
        )

    print(f"\nSaved vector store info to: {OUTPUT_JSON}")
    print(f"Prepared docs saved in: {PREPARED_DIR}")


if __name__ == "__main__":
    main()