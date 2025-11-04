import streamlit as st
import tempfile
import os
import shutil
from pathlib import Path
import subprocess
from typing import Dict, List, Tuple
import requests

# ----------------------------- Theme Toggle -----------------------------

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

st.set_page_config(page_title="Codebase Genius — Streamlit Frontend", layout="wide", initial_sidebar_state="expanded")

# One-click light/dark mode toggle button
if st.button("🌙 Dark Mode" if not st.session_state.dark_mode else "☀️ Light Mode"):
    toggle_theme()

page_bg = "#0E1117" if st.session_state.dark_mode else "#FFFFFF"
text_color = "#FFFFFF" if st.session_state.dark_mode else "#000000"

st.markdown(f"""
    <style>
    body {{ background-color: {page_bg}; color: {text_color}; }}
    .stApp {{ background-color: {page_bg}; color: {text_color}; }}
    </style>
""", unsafe_allow_html=True)

# ----------------------------- App Description -----------------------------

st.title("Codebase Genius — Streamlit Frontend")
st.markdown(
    """
This app reads and documents codebases in **multiple programming languages**. It clones repositories,
parses source files (Python, JavaScript, Java, C/C++, Go, Rust, PHP, HTML, CSS, etc.), and sends the
collected source to a documentation generator (usually backed by an LLM). Users only toggle **Use OpenAI for Documentation** and
receive a README-style output.
"""
)

# ----------------------------- Config ------------------------------------
BACKEND_URL = os.getenv("CODEGEN_BACKEND", "http://localhost:8000/generate-docs")

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


# ----------------------------- Documentation call ------------------------

def generate_documentation_backend(content: str, output_style: str = "readme_full") -> Tuple[bool, str]:
    payload = {
        "content": content,
        "output_style": output_style,
    }
    try:
        resp = requests.post(BACKEND_URL, json=payload, timeout=180)
    except requests.exceptions.RequestException as e:
        return False, f"Failed to connect to documentation backend: {e}"

    if resp.status_code != 200:
        return False, f"Backend error {resp.status_code}: {resp.text}"

    try:
        j = resp.json()
        doc = j.get("documentation") or j.get("doc") or j.get("result")
        if not doc:
            return False, "Backend returned success but no documentation payload."
        return True, doc
    except Exception:
        return False, "Backend returned non-JSON response or invalid format."


# ----------------------------- Main UI -----------------------------------

with st.sidebar:
    st.header("Repository Input")
    repo_url = st.text_input("GitHub repo URL or local folder path:")
    tmp_dir = st.text_input("Local clone folder (optional)", value="")

    st.markdown("---")
    st.header("Documentation Options")
    use_openai = st.checkbox("Use OpenAI for Documentation", value=True)
    output_style = st.selectbox("Documentation style", options=[
        ("readme_full", "Full README (detailed overview, install, usage, examples, architecture)"),
        ("readme_short", "Short README (concise overview and usage)"),
        ("api_reference", "API reference (functions/classes list with doc summaries)")
    ], index=0)
    output_style = output_style[0]

    st.markdown("---")
    if st.button("Generate Documentation"):
        st.session_state.start = True

st.subheader("Codebase Genius — Multi-language Analyzer")
if 'start' not in st.session_state:
    st.info("Enter a repository URL or path in the sidebar, then click 'Generate Documentation'.")
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
        files = []
        for f in Path(st.session_state.repo_root).rglob("*"):
            if f.is_file() and f.suffix.lower() not in [".png", ".jpg", ".jpeg", ".gif", ".svg", ".exe", ".dll"]:
                try:
                    text = f.read_text(encoding="utf-8", errors="ignore")
                    files.append({"path": str(f.relative_to(st.session_state.repo_root)), "content": text})
                except Exception:
                    files.append({"path": str(f.relative_to(st.session_state.repo_root)), "content": "<read error>"})

        bundle = {
            "repo_name": Path(st.session_state.repo_root).name,
            "files": files,
        }

        if use_openai:
            st.info("Generating documentation (this may take a moment)...")
            success, doc_or_err = generate_documentation_backend(bundle, output_style=output_style)
            if not success:
                st.error(doc_or_err)
                st.info("Make sure your documentation backend is running and reachable at the configured endpoint.")
            else:
                st.subheader("Generated Documentation (README-style)")
                st.markdown(doc_or_err)
                st.download_button("Download docs.md", data=doc_or_err, file_name=f"{bundle['repo_name']}_README.md")
        else:
            st.warning("OpenAI documentation is disabled. Toggle 'Use OpenAI for Documentation' to enable.")

        st.markdown("### Done — file tree displayed and documentation processing complete.")