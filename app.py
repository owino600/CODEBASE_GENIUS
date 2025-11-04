import streamlit as st
import tempfile
import os
import shutil
from pathlib import Path
import subprocess
from typing import Dict, List, Tuple
import requests

st.set_page_config(page_title="Codebase Genius — Streamlit Frontend", layout="wide")

st.title("Codebase Genius — Streamlit Frontend")
st.markdown(
    """
This app reads and documents codebases in **multiple programming languages**. It clones repositories,
parses source files (Python, JavaScript, Java, C/C++, Go, Rust, PHP, HTML, CSS, etc.), and generates
structured Markdown documentation summarizing the project's logic, components, and relationships.
"""
)

# ----------------------------- Utilities ---------------------------------

def run_cmd(cmd, cwd=None, timeout=60):
    try:
        p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return "<timeout>"


@st.cache_data(show_spinner=False)
def clone_repo(url: str, dest: str) -> Tuple[bool, str]:
    if os.path.exists(dest):
        shutil.rmtree(dest)
    cmd = f"git clone --depth 1 {url} {dest}"
    out = run_cmd(cmd)
    success = os.path.exists(dest) and any(Path(dest).rglob("*"))
    return success, out


@st.cache_data
def build_file_tree(root: str, ignore: List[str] = None) -> Dict:
    if ignore is None:
        ignore = [".git", "node_modules", "venv", "env", "__pycache__"]
    root_path = Path(root)
    tree = {}
    for p in sorted(root_path.rglob("*")):
        parts = p.relative_to(root_path).parts
        if any(i in parts for i in ignore):
            continue
        cursor = tree
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor.get(part), dict):
                cursor[part] = {}
            cursor = cursor[part]
        if isinstance(cursor, dict):
            cursor[parts[-1]] = {} if p.is_dir() else None
    return tree


def render_tree(tree: Dict, depth=0) -> str:
    s = ""
    for k, v in tree.items():
        s += "  " * depth + f"- {k}\n"
        if isinstance(v, dict):
            s += render_tree(v, depth + 1)
    return s


# ----------------------------- Documentation Generator -----------------------------------

def generate_documentation_backend(content: str, backend_url: str) -> str:
    response = requests.post(
        backend_url,
        json={"content": content},
        headers={"Content-Type": "application/json"},
        timeout=180
    )
    if response.status_code == 200:
        return response.json().get("documentation", "No documentation generated.")
    return f"Error: {response.status_code} - {response.text}"


# ----------------------------- Main UI -----------------------------------

with st.sidebar:
    st.header("Repository Input")
    repo_url = st.text_input("GitHub repo URL or local folder path:")
    tmp_dir = st.text_input("Local clone folder (optional)", value="")

    st.header("Options & Backend Integration")
    use_backend = st.checkbox("Use Backend for Documentation", value=True)
    backend_url = st.text_input("Backend API Endpoint", value="http://localhost:8000/generate-docs") if use_backend else None

    st.markdown("---")
    if st.button("Process Repository"):
        st.session_state.start = True

st.subheader("Codebase Genius — Multi-language Analyzer")
if 'start' not in st.session_state:
    st.info("Enter a repository URL or path in the sidebar, then click 'Process Repository'.")
else:
    dest = tmp_dir or tempfile.mkdtemp(prefix="codegen_")
    if repo_url.startswith("http"):
        st.write(f"Cloning into: {dest}")
        ok, out = clone_repo(repo_url, dest)
        if not ok:
            st.error("Failed to clone repository.")
            st.code(out)
        else:
            st.success("Repository cloned.")
            st.session_state.repo_root = dest
    else:
        if os.path.exists(repo_url):
            st.session_state.repo_root = repo_url
        else:
            st.error("Invalid path provided.")

    if 'repo_root' in st.session_state:
        st.markdown("### File Tree")
        tree = build_file_tree(st.session_state.repo_root)
        st.text_area("Structure", value=render_tree(tree), height=300)

        st.markdown("### Documentation Generator")
        all_content = "\n\n".join(
            open(f, encoding="utf-8", errors="ignore").read()
            for f in Path(st.session_state.repo_root).rglob("*") if f.is_file()
        )

        if use_backend and backend_url:
            st.info("Generating documentation through backend...")
            doc = generate_documentation_backend(all_content, backend_url)
            st.subheader("Generated Documentation")
            st.markdown(doc)
        else:
            st.warning("Backend integration disabled. Please enable it in the sidebar.")

        st.markdown("### Done — File tree and backend integration restored successfully.")