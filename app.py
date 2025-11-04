import streamlit as st
import tempfile
import os
import shutil
from pathlib import Path
import subprocess
import networkx as nx
import matplotlib.pyplot as plt
import ast
import textwrap
from typing import Dict, List, Tuple, Optional

# Optional LLM support
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="Codebase Genius — Streamlit Frontend", layout="wide")

st.title("Codebase Genius — Streamlit Frontend")
st.markdown(
    """
A single-file Streamlit frontend for the *Codebase Genius* assignment. This app is
built to be run locally and can operate standalone (clone & parse repos) or talk to
an external Jac/byLLM backend if you have one running.

**Features included:**
- Clone a public GitHub repository (or point to a local folder)
- Generate a file-tree explorer
- Summarise README (local heuristic or OpenAI if API key provided)
- Parse Python files to build a lightweight Code Context Graph (CCG)
- Visualise relationships between modules, classes and functions
- Preview files and export generated markdown documentation
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
    """Clone a repository into dest. Returns (success, message)."""
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
        # build nested dict
        cursor = tree
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        name = parts[-1]
        cursor.setdefault(name, None)
    return tree


def render_tree(tree: Dict, depth=0) -> str:
    s = ""
    for k, v in tree.items():
        s += "  " * depth + f"- {k}\n"
        if isinstance(v, dict):
            s += render_tree(v, depth + 1)
    return s


# ----------------- README summariser (simple + optional LLM) ---------------

def read_readme(root: str) -> Optional[str]:
    candidates = ["README.md", "readme.md", "README.rst", "README.txt"]
    for c in candidates:
        p = Path(root) / c
        if p.exists():
            return p.read_text(encoding="utf-8", errors="ignore")
    # look for any file with README-like
    for p in Path(root).iterdir():
        if p.is_file() and "readme" in p.name.lower():
            return p.read_text(encoding="utf-8", errors="ignore")
    return None


def heuristic_summary(text: str, max_chars: int = 800) -> str:
    if not text:
        return "No README found."
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # try to pick first paragraphs
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    out = "\n\n".join(paras[:3])
    return out[:max_chars] + "..."


def openai_summary(text: str, api_key: str, max_tokens: int = 300) -> str:
    if not OPENAI_AVAILABLE:
        return "OpenAI library not installed. Install `openai` to enable LLM summarisation."
    openai.api_key = api_key
    prompt = (
        "Summarise the following project README into a concise overview (4-6 sentences).\n\n" + text
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI summarisation failed: {e}"


# ----------------- Code Analyzer (AST-based lightweight CCG) -------------

class CodeAnalyzer:
    def __init__(self, root: str):
        self.root = Path(root)
        self.modules = {}  # path -> parsed AST
        self.entities = {}  # node id -> info
        self.edges = []  # (from, to, label)

    def discover_python_files(self) -> List[Path]:
        return [p for p in self.root.rglob("*.py") if ".venv" not in p.parts]

    def parse(self):
        py_files = self.discover_python_files()
        for p in py_files:
            try:
                src = p.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(src)
                self.modules[str(p.relative_to(self.root))] = tree
            except Exception:
                continue
        self._extract_entities()
        self._extract_calls()

    def _extract_entities(self):
        for mod_path, tree in self.modules.items():
            visitor = _EntityVisitor(mod_path)
            visitor.visit(tree)
            for nid, info in visitor.entities.items():
                self.entities[nid] = info

    def _extract_calls(self):
        for mod_path, tree in self.modules.items():
            visitor = _CallVisitor(mod_path)
            visitor.visit(tree)
            for call_from, call_to in visitor.calls:
                # call_from and call_to are (module, name)
                from_id = f"{call_from[0]}::{call_from[1]}"
                to_id = f"{call_to[0]}::{call_to[1]}"
                self.edges.append((from_id, to_id, "calls"))

    def build_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        # add nodes
        for nid, info in self.entities.items():
            G.add_node(nid, **info)
        for a, b, lbl in self.edges:
            if not G.has_node(a):
                G.add_node(a, label=a)
            if not G.has_node(b):
                G.add_node(b, label=b)
            G.add_edge(a, b, label=lbl)
        return G


class _EntityVisitor(ast.NodeVisitor):
    def __init__(self, mod_path):
        self.mod = mod_path
        self.entities = {}

    def visit_FunctionDef(self, node: ast.FunctionDef):
        nid = f"{self.mod}::{node.name}"
        self.entities[nid] = {
            "type": "function",
            "name": node.name,
            "lineno": node.lineno,
        }
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        nid = f"{self.mod}::{node.name}"
        bases = [ast.unparse(b) if hasattr(ast, 'unparse') else getattr(b, 'id', '') for b in node.bases]
        self.entities[nid] = {
            "type": "class",
            "name": node.name,
            "lineno": node.lineno,
            "bases": bases,
        }
        self.generic_visit(node)


class _CallVisitor(ast.NodeVisitor):
    def __init__(self, mod_path):
        self.mod = mod_path
        self.current_fn = None
        self.calls = []  # list of ((module, from_name), (module, to_name))

    def visit_FunctionDef(self, node: ast.FunctionDef):
        prev = self.current_fn
        self.current_fn = node.name
        self.generic_visit(node)
        self.current_fn = prev

    def visit_Call(self, node: ast.Call):
        # We attempt to get a simple name for called function
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        if func_name and self.current_fn:
            from_ = (self.mod, self.current_fn)
            to_ = (self.mod, func_name)
            self.calls.append((from_, to_))
        self.generic_visit(node)


# ----------------- UI Layout ---------------------------------------------

with st.sidebar:
    st.header("Repository input")
    repo_url = st.text_input("GitHub repo URL (https://github.com/owner/repo) or local folder:")
    tmp_dir = st.text_input("Local clone folder (optional)", value="")

    st.markdown("---")
    st.header("Options & LLM")
    use_openai = st.checkbox("Use OpenAI for README summarisation (optional)")
    if use_openai:
        api_key = st.text_input("OpenAI API key", type="password")
    else:
        api_key = None

    show_hidden = st.checkbox("Show hidden files (like .env)", value=False)

    st.markdown("---")
    st.write("App controls")
    if st.button("Process repository"):
        st.session_state.start = True


col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Repository Explorer")
    if 'start' not in st.session_state:
        st.info("Enter a GitHub URL or a local path in the sidebar, then click 'Process repository'.")
    else:
        if repo_url.startswith("http"):
            dest = tmp_dir or tempfile.mkdtemp(prefix="codegen_")
            progress = st.progress(0)
            st.write(f"Cloning into: {dest}")
            ok, out = clone_repo(repo_url, dest)
            progress.progress(50)
            if not ok:
                st.error("Failed to clone repository. See output for details.")
                st.code(out)
            else:
                st.success("Repository cloned.")
                st.session_state.repo_root = dest
                tree = build_file_tree(dest)
                st.text_area("File tree (collapsed)", value=render_tree(tree), height=300)
                progress.progress(80)
                progress.empty()
        else:
            # treat as local folder
            if os.path.exists(repo_url):
                st.session_state.repo_root = repo_url
                tree = build_file_tree(repo_url)
                st.text_area("File tree (collapsed)", value=render_tree(tree), height=300)
            else:
                st.error("Local path not found.")

    st.markdown("---")
    st.subheader("README Summary")
    if 'repo_root' in st.session_state:
        readme = read_readme(st.session_state.repo_root)
        if readme:
            if use_openai and api_key:
                with st.spinner("Summarising with OpenAI..."):
                    summary = openai_summary(readme, api_key)
            else:
                summary = heuristic_summary(readme)
            st.markdown(summary)
            if st.button("Show full README"):
                st.code(readme[:10000])
        else:
            st.info("No README found in repository root.")

with col2:
    st.subheader("Code Context Graph & Analysis")
    if 'repo_root' in st.session_state:
        analyzer = CodeAnalyzer(st.session_state.repo_root)
        with st.spinner("Parsing Python files and building CCG..."):
            analyzer.parse()
        G = analyzer.build_graph()

        st.write(f"Discovered nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

        # simple interactive filters
        node_type = st.selectbox("Show nodes of type:", options=["all", "function", "class"], index=0)
        to_show = [n for n, d in G.nodes(data=True) if node_type == "all" or d.get("type") == node_type]

        st.write("### Node list (sample)")
        for n in list(to_show)[:40]:
            d = G.nodes[n]
            st.write(f"- **{n}** — {d.get('type', '')} (line: {d.get('lineno','?')})")

        # Draw graph
        fig, ax = plt.subplots(figsize=(10, 6))
        subG = G.subgraph(list(to_show)[:80]).copy()
        pos = nx.spring_layout(subG, seed=42)
        nx.draw_networkx_nodes(subG, pos, node_size=300, ax=ax)
        nx.draw_networkx_edges(subG, pos, ax=ax, arrowsize=12)
        nx.draw_networkx_labels(subG, pos, font_size=8, ax=ax)
        ax.set_axis_off()
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("File preview & export")
        # file browser simple
        py_files = sorted([str(p.relative_to(st.session_state.repo_root)) for p in Path(st.session_state.repo_root).rglob("*.py")])
        sel = st.selectbox("Select a python file to preview", options=["(none)"] + py_files)
        if sel and sel != "(none)":
            full = Path(st.session_state.repo_root) / sel
            st.code(full.read_text(encoding='utf-8', errors='ignore')[:20000])

        if st.button("Export documentation (markdown)"):
            md = build_markdown_doc(st.session_state.repo_root, analyzer)
            out_folder = Path(st.session_state.repo_root) / "outputs" / "codegen-docs"
            out_folder.mkdir(parents=True, exist_ok=True)
            out_path = out_folder / "docs.md"
            out_path.write_text(md, encoding="utf-8")
            st.success(f"Docs exported to {out_path}")
            with open(out_path, 'r', encoding='utf-8') as f:
                st.download_button("Download docs.md", f.read(), file_name="docs.md")


# ----------------- Markdown generator -----------------------------------

def build_markdown_doc(root: str, analyzer: CodeAnalyzer) -> str:
    root_p = Path(root)
    name = root_p.name
    readme = read_readme(root) or ""
    summary = heuristic_summary(readme)
    md = []
    md.append(f"# {name} — Auto-generated Documentation\n")
    md.append("## Project overview\n")
    md.append(summary + "\n\n")
    md.append("## File tree\n")
    md.append("``""\n" + render_tree(build_file_tree(root)) + "\n```\n")
    md.append("## API & Code Context Graph\n")

    G = analyzer.build_graph()
    md.append(f"Discovered {G.number_of_nodes()} entities and {G.number_of_edges()} relationships.\n\n")

    md.append("### Top entities\n")
    for i, (n, d) in enumerate(G.nodes(data=True)):
        if i >= 40:
            break
        md.append(f"- `{n}` — {d.get('type','?')} (line: {d.get('lineno','?')})\n")

    md.append("\n### Relationships (sample)\n")
    for i, (a, b, ed) in enumerate([(u, v, d) for u, v, d in G.edges(data=True)]):
        if i >= 80:
            break
        md.append(f"- `{a}` {ed.get('label','')} `{b}`\n")

    md.append("\n## README (original)\n")
    md.append("````\n" + (readme[:20000] or "(no README)") + "\n````\n")

    md.append("\n## Generated by Codebase Genius (Streamlit frontend)\n")
    return "\n".join(md)


# ----------------- Footer / Tips ----------------------------------------

st.markdown("---")
st.caption("Tip: This frontend is intentionally self-contained. For full multi-agent behaviour implement the Jac backend from the assignment and wire the Streamlit UI to it via HTTP.")

st.markdown("## Quick run instructions")
st.markdown(textwrap.dedent("""
1. Install dependencies: `pip install streamlit gitpython networkx matplotlib openai`
2. Run: `streamlit run streamlit_codebase_genius.py`
3. Enter a public GitHub repo URL or local folder, then click **Process repository**.

Optional: provide an OpenAI API key to enable better README summarisation.
"""))