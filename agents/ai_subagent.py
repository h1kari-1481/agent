#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import fnmatch
import time
import argparse
import logging
import codecs
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# -------------------------
# Defaults and constants
# -------------------------

DEFAULT_INCLUDE = "**/*"
DEFAULT_EXCLUDES = [
    "**/.git/**",
    "**/.hg/**",
    "**/.svn/**",
    "**/node_modules/**",
    "**/.venv/**",
    "**/__pycache__/**",
    "**/dist/**",
    "**/build/**",
    "**/.next/**",
    "**/.cache/**",
    "**/*.min.*",
    "**/analysis/**",  # avoid re-ingesting outputs
]
DEFAULT_EXTENSIONS = [
    ".py", ".ts", ".tsx", ".js", ".jsx",
    ".json", ".yml", ".yaml", ".toml", ".ini",
    ".md", ".rst", ".txt",  # include .txt by default
    ".java", ".kt", ".go", ".rs", ".cpp", ".cc", ".hpp", ".c", ".h",
    ".rb", ".php", ".swift",
    ".sql", ".graphql", ".gql",
    ".sh", ".ps1", ".bat", ".dockerfile", "Dockerfile",
]

DEFAULT_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
DEFAULT_API_BASE = os.getenv("LLM_API_BASE", "https://api.deepseek.com")
DEFAULT_API_KEY = os.getenv("DEEPSEEK_API_KEY", os.getenv("LLM_API_KEY", ""))

BASE_SYSTEM_PROMPT = """You are a senior software analyst. You will be given code files from a software project.
Your task: produce thorough, accurate, concise analysis following the required output format.
- Emphasize architecture, module responsibilities, data flow, external dependencies, risks, TODOs, and test coverage.
- Do not hallucinate. If uncertain, say "unknown".
- Be consistent across files and the project summary.
- When asked to return JSON, return ONLY valid JSON with no extra text.
"""

# Built-in default JSON format for output
DEFAULT_FORMAT_TEMPLATE = {
    "project_overview": "string, a concise summary of what the project does",
    "architecture": {
        "style": "string, e.g., monolith/microservices/layered/CQRS/etc.",
        "layers_or_modules": "string, brief explanation of main layers or modules",
        "data_flow": "string, how data flows across components",
        "key_components": [
            {"name": "string", "role": "string", "key_files": ["string"]}
        ]
    },
    "modules": [
        {
            "name": "string",
            "path_patterns": ["string"],
            "responsibilities": "string",
            "public_apis": ["string"],
            "key_dependencies": ["string"],
            "risks": ["string"],
            "todos": ["string"]
        }
    ],
    "dependencies": {"internal": ["string"], "external": ["string"]},
    "risks": ["string"],
    "testing": {
        "coverage_estimate": "string",
        "test_types": ["string"],
        "gaps": ["string"]
    },
    "recommended_actions": ["string"],
    "files": [
        {
            "path": "string",
            "summary": "string",
            "key_points": ["string"],
            "public_apis": ["string"],
            "dependencies": ["string"],
            "risks": ["string"],
            "todos": ["string"]
        }
    ]
}

FILE_ANALYSIS_TEMPLATE = {
    "path": "string",
    "summary": "string",
    "key_points": ["string"],
    "public_apis": ["string"],
    "dependencies": ["string"],
    "risks": ["string"],
    "todos": ["string"]
}

# -------------------------
# Config
# -------------------------

@dataclass
class AgentConfig:
    root: str
    include: List[str]
    exclude: List[str]
    extensions: List[str]
    max_files: Optional[int]
    max_file_bytes: int
    chunk_size: int
    chunk_overlap: int
    concurrency: int
    model: str
    api_base: str
    api_key: str
    temperature: float
    output: str
    output_format: str  # "json" or "markdown"
    format_template_path: Optional[str]
    dry_run: bool
    timeout: int  # seconds
    lan: str  # "zh" or "en"
    strict_json: bool  # force model to return strict JSON
    summary_output: Optional[str]  # path to write a standalone summary markdown
    append_summary: bool  # append a self-summary at the end of main markdown

# -------------------------
# Language prompt builder
# -------------------------

def build_system_prompt(lan: str) -> str:
    if lan.lower() == "zh":
        lang_block = (
            "\nLanguage requirement:\n"
            "- Use Simplified Chinese for ALL outputs.\n"
            "- 对所有输出使用简体中文，包括 JSON 中的字符串值与 Markdown 文本。\n"
            "- JSON 的字段名保持英文不变，字符串值使用简体中文。\n"
        )
    else:
        lang_block = (
            "\nLanguage requirement:\n"
            "- Use English for ALL outputs.\n"
            "- Keep JSON field names unchanged; string values should be in English.\n"
        )
    return BASE_SYSTEM_PROMPT + lang_block

def _mask_secret(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    if len(s) <= 8:
        return "***"
    return f"{s[:4]}...{s[-4:]}"

# -------------------------
# Encoding and file utils
# -------------------------

def _detect_text_encoding(sample: bytes) -> Optional[str]:
    # Recognize common BOMs
    if sample.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if sample.startswith(codecs.BOM_UTF16_LE):
        return "utf-16-le"
    if sample.startswith(codecs.BOM_UTF16_BE):
        return "utf-16-be"
    # Try common encodings
    for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            sample.decode(enc)
            return enc
        except Exception:
            continue
    return None

def is_binary_file(path: Path, sniff_bytes: int = 2048) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(sniff_bytes)
        # If recognized as text, it's not binary
        enc = _detect_text_encoding(chunk)
        if enc:
            return False
        # Heuristic fallback
        if b"\x00" in chunk:
            return True
        text_bytes = sum(32 <= b <= 127 or b in (9, 10, 13) for b in chunk)
        if len(chunk) == 0:
            return False
        return (text_bytes / len(chunk)) < 0.7
    except Exception:
        return True

def read_text_safely(path: Path, max_bytes: int) -> str:
    try:
        with path.open("rb") as f:
            raw = f.read(max_bytes + 1)
        truncated = raw[:max_bytes]
        enc = _detect_text_encoding(truncated) or "utf-8"
        text = truncated.decode(enc, errors="replace")
        if len(raw) > max_bytes:
            text += "\n\n/* [TRUNCATED for size limit] */\n"
        return text
    except Exception as e:
        return f"/* [ERROR reading file: {e}] */"

def normalize_globs(values: List[str]) -> List[str]:
    out: List[str] = []
    for v in values:
        if not v:
            continue
        parts = [p.strip() for p in v.split(",") if p.strip()]
        out.extend(parts)
    return out

def filter_files(root: Path, includes: List[str], excludes: List[str], extensions: List[str]) -> List[Path]:
    # Expand include patterns
    candidates_set = set()
    if not includes:
        includes = [DEFAULT_INCLUDE]
    for pattern in includes:
        for p in root.glob(pattern):
            if p.is_file():
                candidates_set.add(p.resolve())

    # Filter by extension
    if extensions:
        extset = set(e.lower() for e in extensions)
        candidates = [Path(p) for p in candidates_set if (Path(p).suffix.lower() in extset or Path(p).name in extset)]
    else:
        candidates = [Path(p) for p in candidates_set]

    # Apply excludes (glob-like)
    def matches_exclude(path: Path) -> bool:
        rel = path.relative_to(root).as_posix()
        for ex in excludes:
            if fnmatch.fnmatch(rel, ex) or fnmatch.fnmatch(path.as_posix(), ex):
                return True
        return False

    filtered = [p for p in candidates if not matches_exclude(p)]
    filtered.sort()
    return filtered

def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    if max_chars <= 0:
        return [text]
    chunks = []
    i = 0
    n = len(text)
    step = max(1, max_chars - overlap)
    while i < n:
        chunks.append(text[i:i + max_chars])
        i += step
    return chunks

# -------------------------
# JSON parsing hardening
# -------------------------

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    m = re.match(r"^```(?:json)?\s*(.+?)\s*```$", s, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    return s

def _extract_braced_json(s: str) -> str:
    start = s.find("{")
    if start == -1:
        return s
    depth = 0
    in_str = False
    esc = False
    for idx in range(start, len(s)):
        ch = s[idx]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:idx + 1]
    return s[start:] if start >= 0 else s

def _cleanup_json_common(s: str) -> str:
    s = re.sub(r",\s*([}\]])", r"\1", s)  # remove trailing commas
    s = re.sub(r"\bNaN\b|\bInfinity\b|\b-?Infinity\b|\bNone\b", "null", s)
    return s

def safe_parse_json(text: str) -> Any:
    cleaned = _strip_code_fences(text)
    cleaned = _extract_braced_json(cleaned)
    cleaned = _cleanup_json_common(cleaned)
    return json.loads(cleaned)

# -------------------------
# LLM Client (DeepSeek via OpenAI-compatible)
# -------------------------

class LLMClient:
    def __init__(self, api_base: str, api_key: str, model: str, timeout: int = 60):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, strict_json: bool = False) -> str:
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=False,
            timeout=self.timeout,
        )
        if strict_json:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        try:
            return resp.choices[0].message.content or ""
        except Exception:
            # Fallback to raw dump
            return json.dumps(resp.model_dump(mode="python"), ensure_ascii=False)

# -------------------------
# Analyzer
# -------------------------

class ProjectAnalyzer:
    def __init__(self, llm: LLMClient, config: AgentConfig,
                 project_format_template: Dict[str, Any],
                 file_format_template: Dict[str, Any]):
        self.llm = llm
        self.cfg = config
        self.project_format_template = project_format_template
        self.file_format_template = file_format_template

    # ---- per-file ----

    def analyze_file(self, path: Path, content: str) -> Dict[str, Any]:
        chunks = chunk_text(content, self.cfg.chunk_size, self.cfg.chunk_overlap)
        chunk_analyses: List[str] = []

        for idx, ch in enumerate(chunks, 1):
            user_prompt = self._build_file_chunk_prompt(path, ch, idx, len(chunks))
            messages = [
                {"role": "system", "content": build_system_prompt(self.cfg.lan)},
                {"role": "user", "content": user_prompt},
            ]
            if self.cfg.dry_run:
                logging.info(f"[DRY RUN] Would analyze chunk {idx}/{len(chunks)} for {path}")
                chunk_analyses.append(f"/* DRY_RUN chunk {idx} summary for {path} */")
            else:
                try:
                    out = self.llm.chat(messages, temperature=self.cfg.temperature, strict_json=False)
                    chunk_analyses.append(out.strip())
                except Exception as e:
                    logging.warning(f"LLM error analyzing chunk {idx}/{len(chunks)} for {path}: {e}")
                    chunk_analyses.append(f"/* ERROR: {e} */")
            time.sleep(0.03)

        merge_prompt = self._build_file_merge_prompt(path, chunk_analyses)
        messages = [
            {"role": "system", "content": build_system_prompt(self.cfg.lan)},
            {"role": "user", "content": merge_prompt},
        ]
        if self.cfg.dry_run:
            return {
                "path": str(path),
                "summary": "DRY_RUN summary",
                "key_points": ["DRY_RUN"],
                "public_apis": [],
                "dependencies": [],
                "risks": [],
                "todos": []
            }
        try:
            out = self.llm.chat(messages, temperature=self.cfg.temperature, strict_json=True)
            parsed = safe_parse_json(out)
            if isinstance(parsed, dict):
                parsed.setdefault("path", str(path))
                return parsed
            else:
                return {
                    "path": str(path),
                    "summary": str(parsed)[:2000],
                    "key_points": [],
                    "public_apis": [],
                    "dependencies": [],
                    "risks": [],
                    "todos": []
                }
        except Exception as e:
            # dump raw
            try:
                debug_path = Path(self.cfg.output).with_suffix(".llm-file-merge-raw.txt")
                with open(debug_path, "a", encoding="utf-8") as fp:
                    fp.write(f"\n\n## {path}\n")
                    fp.write(out if isinstance(out, str) else str(out))
                logging.error(f"Saved raw file-merge output to: {debug_path}")
            except Exception:
                pass
            logging.error(f"LLM error merging file analysis for {path}: {e}")
            return {
                "path": str(path),
                "summary": f"/* ERROR merging analysis: {e} */",
                "key_points": [],
                "public_apis": [],
                "dependencies": [],
                "risks": [],
                "todos": []
            }

    # ---- project synthesis ----

    def analyze_project(self, file_analyses: List[Dict[str, Any]], project_name: str) -> Dict[str, Any]:
        comp = self._compress_file_analyses_for_project(file_analyses)
        prompt = self._build_project_synthesis_prompt(project_name, comp)
        messages = [
            {"role": "system", "content": build_system_prompt(self.cfg.lan)},
            {"role": "user", "content": prompt},
        ]
        if self.cfg.dry_run:
            return {
                "project_overview": f"DRY_RUN overview for {project_name}",
                "architecture": {
                    "style": "unknown",
                    "layers_or_modules": "unknown",
                    "data_flow": "unknown",
                    "key_components": []
                },
                "modules": [],
                "dependencies": {"internal": [], "external": []},
                "risks": [],
                "testing": {"coverage_estimate": "unknown", "test_types": [], "gaps": []},
                "recommended_actions": [],
                "files": file_analyses,
            }
        try:
            out = self.llm.chat(messages, temperature=self.cfg.temperature, strict_json=True)
            parsed = safe_parse_json(out)
            if isinstance(parsed, dict):
                parsed.setdefault("files", file_analyses)
                return parsed
            else:
                return {
                    "project_overview": "See raw_text",
                    "raw_text": str(parsed)[:20000],
                    "files": file_analyses
                }
        except Exception as e:
            # dump raw
            try:
                debug_path = Path(self.cfg.output).with_suffix(".llm-project-raw.txt")
                with open(debug_path, "w", encoding="utf-8") as fp:
                    fp.write(out if isinstance(out, str) else str(out))
                logging.error(f"Saved raw project synthesis output to: {debug_path}")
            except Exception:
                pass
            logging.error(f"LLM error in project synthesis: {e}")
            return {
                "project_overview": f"/* ERROR project synthesis: {e} */",
                "files": file_analyses
            }

    # ---- prompt builders ----

    def _build_file_chunk_prompt(self, path: Path, chunk: str, idx: int, total: int) -> str:
        if self.cfg.lan.lower() == "zh":
            header = (
                f"你正在分析文件: {path}\n"
                f"分片 {idx}/{total}。请用简体中文，给出简明要点，涵盖：\n"
                f"- 目的与核心职责\n"
                f"- 对外 API（函数、类、接口、端点）及简述\n"
                f"- 重要依赖（内部模块、外部库）\n"
                f"- 风险（安全、性能、可维护性）\n"
                f"- TODO/ FIXME\n\n"
                f"---BEGIN FILE CONTENT CHUNK---\n{chunk}\n---END FILE CONTENT CHUNK---"
            )
        else:
            header = (
                f"You are analyzing file: {path}\n"
                f"Chunk {idx}/{total}. Provide a concise analysis in plain text, focusing on:\n"
                f"- Purpose and key responsibilities\n"
                f"- Public APIs (functions, classes, endpoints) with brief descriptions\n"
                f"- Notable dependencies (internal modules, external libraries)\n"
                f"- Risks (security, performance, maintainability)\n"
                f"- TODOs or FIXMEs\n\n"
                f"---BEGIN FILE CONTENT CHUNK---\n{chunk}\n---END FILE CONTENT CHUNK---"
            )
        return header

    def _build_file_merge_prompt(self, path: Path, chunk_analyses: List[str]) -> str:
        lang_line = "- 所有字符串值请使用简体中文，字段名保持英文不变。\n" if self.cfg.lan.lower() == "zh" \
            else "- Use English for all string values; keep JSON field names in English.\n"
        return (
            "You analyzed multiple chunks of a single file. Now merge them into one JSON object strictly matching this template:\n"
            f"{json.dumps(self.file_format_template, ensure_ascii=False, indent=2)}\n\n"
            "- Output STRICT JSON only. No markdown, no explanations, no code fences.\n"
            "- Ensure 'path' is the file path string.\n"
            "- Do not include fields not in the template.\n"
            "- Use arrays where arrays are expected.\n"
            "- Use 'unknown' or empty arrays where not applicable.\n"
            f"{lang_line}\n"
            f"File path: {path}\n"
            "Chunk analyses to merge:\n"
            + "\n\n".join([f"--- CHUNK ANALYSIS {i+1} ---\n{a}" for i, a in enumerate(chunk_analyses)])
        )

    def _compress_file_analyses_for_project(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compact = []
        for f in files:
            compact.append({
                "path": f.get("path", "unknown"),
                "summary": f.get("summary", "")[:800],
                "public_apis": f.get("public_apis", [])[:20],
                "dependencies": f.get("dependencies", [])[:20],
                "risks": f.get("risks", [])[:10],
                "todos": f.get("todos", [])[:10],
            })
        return compact

    def _build_project_synthesis_prompt(self, project_name: str, compact_files: List[Dict[str, Any]]) -> str:
        lang_line = "- 所有字符串值请使用简体中文，字段名保持英文不变。\n" if self.cfg.lan.lower() == "zh" \
            else "- Use English for all string values; keep JSON field names in English.\n"
        return (
            f"Project: {project_name}\n"
            "Produce ONE JSON object strictly matching this template:\n"
            f"{json.dumps(self.project_format_template, ensure_ascii=False, indent=2)}\n\n"
            "- Output STRICT JSON only. No markdown, no explanations, no code fences.\n"
            "- Use the 'files' field to include the finalized per-file analyses (ensure each item conforms to the template if present).\n"
            "- Derive higher-level 'architecture', 'modules', 'dependencies', 'risks', 'testing', and 'recommended_actions' from the files.\n"
            "- Be consistent and avoid hallucination. Use 'unknown' if needed.\n"
            f"{lang_line}\n"
            "Here are the per-file compact analyses:\n"
            + json.dumps(compact_files, ensure_ascii=False, indent=2)
        )

# -------------------------
# Rendering
# -------------------------

def render_output(result: Dict[str, Any], output_format: str, lan: str) -> str:
    if output_format.lower() == "json":
        # add self_summary textual block into JSON too for completeness
        result = dict(result)
        result["self_summary_text"] = build_self_summary_md(result, lan=lan)
        return json.dumps(result, ensure_ascii=False, indent=2)
    elif output_format.lower() == "markdown":
        return render_markdown(result, lan=lan)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

def render_markdown(result: Dict[str, Any], lan: str = "zh") -> str:
    md = []
    po = result
    zh = lan.lower() == "zh"

    def h2(title): md.append(f"## {title}\n")
    def h3(title): md.append(f"### {title}\n")

    md.append("# 项目分析\n\n" if zh else "# Project Analysis\n\n")

    if "project_overview" in po:
        h2("概览" if zh else "Overview")
        md.append(f"{po.get('project_overview','')}\n\n")

    if "architecture" in po:
        arch = po.get("architecture", {})
        h2("架构" if zh else "Architecture")
        md.append(f"{'- 风格: ' if zh else '- Style: '}{arch.get('style','unknown')}\n")
        md.append(f"{'- 分层/模块: ' if zh else '- Layers/Modules: '}{arch.get('layers_or_modules','')}\n")
        md.append(f"{'- 数据流: ' if zh else '- Data Flow: '}{arch.get('data_flow','')}\n")
        kcs = arch.get("key_components", [])
        if kcs:
            md.append("- 关键组件:\n" if zh else "- Key Components:\n")
            for kc in kcs:
                if zh:
                    md.append(f"  - {kc.get('name','unknown')}: {kc.get('role','')}, 文件: {', '.join(kc.get('key_files', []))}\n")
                else:
                    md.append(f"  - {kc.get('name','unknown')}: {kc.get('role','')}, files: {', '.join(kc.get('key_files', []))}\n")
        md.append("\n")

    if "modules" in po and po["modules"]:
        h2("模块" if zh else "Modules")
        for m in po["modules"]:
            h3(m.get("name", "unknown"))
            if m.get("path_patterns"):
                md.append(f"{'- 路径: ' if zh else '- Paths: '}{', '.join(m['path_patterns'])}\n")
            md.append(f"{'- 职责: ' if zh else '- Responsibilities: '}{m.get('responsibilities','')}\n")
            if m.get("public_apis"):
                md.append(f"{'- 对外 API: ' if zh else '- Public APIs: '}{', '.join(m['public_apis'])}\n")
            if m.get("key_dependencies"):
                md.append(f"{'- 关键依赖: ' if zh else '- Key Dependencies: '}{', '.join(m['key_dependencies'])}\n")
            if m.get("risks"):
                md.append(f"{'- 风险: ' if zh else '- Risks: '}{', '.join(m['risks'])}\n")
            if m.get("todos"):
                md.append(f"{'- 待办: ' if zh else '- TODOs: '}{', '.join(m['todos'])}\n")
            md.append("\n")

    if "dependencies" in po:
        dep = po["dependencies"]
        h2("依赖" if zh else "Dependencies")
        if dep.get("internal"):
            md.append(f"{'- 内部: ' if zh else '- Internal: '}{', '.join(dep['internal'])}\n")
        if dep.get("external"):
            md.append(f"{'- 外部: ' if zh else '- External: '}{', '.join(dep['external'])}\n")
        md.append("\n")

    if "risks" in po and po["risks"]:
        h2("风险" if zh else "Risks")
        for r in po["risks"]:
            md.append(f"- {r}\n")
        md.append("\n")

    if "testing" in po:
        t = po["testing"]
        h2("测试" if zh else "Testing")
        md.append(f"{'- 覆盖率: ' if zh else '- Coverage: '}{t.get('coverage_estimate','unknown')}\n")
        if t.get("test_types"):
            md.append(f"{'- 类型: ' if zh else '- Types: '}{', '.join(t['test_types'])}\n")
        if t.get("gaps"):
            md.append(f"{'- 缺口: ' if zh else '- Gaps: '}{', '.join(t['gaps'])}\n")
        md.append("\n")

    if "recommended_actions" in po and po["recommended_actions"]:
        h2("建议动作" if zh else "Recommended Actions")
        for a in po["recommended_actions"]:
            md.append(f"- {a}\n")
        md.append("\n")

    if "files" in po and po["files"]:
        h2("文件" if zh else "Files")
        for f in po["files"]:
            h3(f.get("path", "unknown"))
            md.append(f"{'- 摘要: ' if zh else '- Summary: '}{f.get('summary','')}\n")
            if f.get("key_points"):
                md.append(f"{'- 要点: ' if zh else '- Key Points: '}{', '.join(f['key_points'])}\n")
            if f.get("public_apis"):
                md.append(f"{'- 对外 API: ' if zh else '- Public APIs: '}{', '.join(f['public_apis'])}\n")
            if f.get("dependencies"):
                md.append(f"{'- 依赖: ' if zh else '- Dependencies: '}{', '.join(f['dependencies'])}\n")
            if f.get("risks"):
                md.append(f"{'- 风险: ' if zh else '- Risks: '}{', '.join(f['risks'])}\n")
            if f.get("todos"):
                md.append(f"{'- 待办: ' if zh else '- TODOs: '}{', '.join(f['todos'])}\n")
            md.append("\n")

    return "".join(md)

# -------------------------
# Self-summary (deterministic)
# -------------------------

def _top_items(items: List[str], n: int = 10) -> List[Tuple[str, int]]:
    freq: Dict[str, int] = {}
    for it in items:
        key = (it or "").strip()
        if not key:
            continue
        freq[key] = freq.get(key, 0) + 1
    return sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:n]

def build_self_summary_md(result: Dict[str, Any], lan: str = "zh") -> str:
    zh = lan.lower() == "zh"
    files = result.get("files", []) or []
    modules = result.get("modules", []) or []
    deps = result.get("dependencies", {}) or {}
    arch = result.get("architecture", {}) or {}
    risks_agg = result.get("risks", []) or []
    recs = result.get("recommended_actions", []) or []
    testing = result.get("testing", {}) or {}

    # Aggregate across files
    file_risks: List[str] = []
    file_todos: List[str] = []
    file_deps: List[str] = []
    for f in files:
        file_risks.extend(f.get("risks", []) or [])
        file_todos.extend(f.get("todos", []) or [])
        file_deps.extend(f.get("dependencies", []) or [])

    top_file_risks = _top_items(file_risks, n=10)
    top_file_todos = _top_items(file_todos, n=10)
    top_file_deps = _top_items(file_deps, n=10)

    # Compose markdown
    lines: List[str] = []
    lines.append("## 文档自我总结\n" if zh else "## Document Self Summary\n")

    # Scope and coverage
    lines.append(f"- {'分析文件数' if zh else 'Files analyzed'}: {len(files)}\n")
    lines.append(f"- {'架构风格' if zh else 'Architecture style'}: {arch.get('style','unknown')}\n")

    # Modules and components
    if modules:
        mod_names = [m.get("name", "unknown") for m in modules][:10]
        lines.append(f"- {'主要模块' if zh else 'Top modules'}: {', '.join(mod_names)}\n")
    kcs = arch.get("key_components", []) or []
    if kcs:
        kc_names = [kc.get("name", "unknown") for kc in kcs][:10]
        lines.append(f"- {'关键组件' if zh else 'Key components'}: {', '.join(kc_names)}\n")

    # External deps
    ext = deps.get("external", []) or []
    if ext or top_file_deps:
        shown_ext = ext[:10] if ext else [d for d, _ in top_file_deps[:10]]
        lines.append(f"- {'外部依赖' if zh else 'External dependencies'}: {', '.join(shown_ext)}\n")

    # Risks
    if risks_agg or top_file_risks:
        lines.append(f"- {'主要风险' if zh else 'Top risks'}:\n")
        if risks_agg:
            for r in risks_agg[:5]:
                lines.append(f"  - {r}\n")
        elif top_file_risks:
            for r, c in top_file_risks[:5]:
                lines.append(f"  - {r} ({c})\n")

    # Testing and gaps
    if testing:
        cov = testing.get("coverage_estimate", "unknown")
        gaps = testing.get("gaps", []) or []
        lines.append(f"- {'测试覆盖率' if zh else 'Test coverage'}: {cov}\n")
        if gaps:
            lines.append(f"- {'测试缺口' if zh else 'Test gaps'}: {', '.join(gaps[:8])}\n")

    # TODOs
    if top_file_todos:
        lines.append(f"- {'跨文件共性 TODO' if zh else 'Cross-file common TODOs'}:\n")
        for t, c in top_file_todos[:5]:
            lines.append(f"  - {t} ({c})\n")

    # Recommended actions
    if recs:
        lines.append(f"- {'建议动作' if zh else 'Recommended actions'}:\n")
        for a in recs[:8]:
            lines.append(f"  - {a}\n")

    # How to use this doc
    if zh:
        lines.append("\n> 说明：以上总结基于对各文件分析结果与项目级综合的再汇总，旨在快速定位核心模块、关键依赖与主要风险，便于制定后续优化与测试计划。\n")
    else:
        lines.append("\n> Note: This self-summary aggregates per-file and project-level findings to highlight key modules, dependencies, and risks, facilitating next-step planning.\n")

    return "".join(lines)

# -------------------------
# Main flow
# -------------------------

def run_analysis(cfg: AgentConfig) -> Dict[str, Any]:
    root = Path(cfg.root).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Root path not found or not a directory: {root}")

    includes = normalize_globs(cfg.include) or [DEFAULT_INCLUDE]
    excludes = normalize_globs(cfg.exclude) or DEFAULT_EXCLUDES
    extensions = [e.strip() for e in cfg.extensions] if cfg.extensions else []

    files = filter_files(root, includes, excludes, extensions if extensions else DEFAULT_EXTENSIONS)
    files = [f for f in files if not is_binary_file(f)]
    if cfg.max_files and cfg.max_files > 0:
        files = files[: cfg.max_files]

    logging.info(f"Discovered {len(files)} files to analyze")

    # If no files, skip LLM calls and return a minimal structure
    if not files:
        overview_msg = (
            f"未在目录下发现可分析文件：{str(root)}。请调整 --extensions/--include/--exclude 后重试。"
            if cfg.lan.lower() == "zh"
            else f"No files discovered under: {str(root)}. Adjust --extensions/--include/--exclude and retry."
        )
        return {
            "project_overview": overview_msg,
            "architecture": {"style": "unknown", "layers_or_modules": "unknown", "data_flow": "unknown", "key_components": []},
            "modules": [],
            "dependencies": {"internal": [], "external": []},
            "risks": [],
            "testing": {"coverage_estimate": "unknown", "test_types": [], "gaps": []},
            "recommended_actions": [],
            "files": []
        }

    # Prepare templates
    if cfg.format_template_path:
        with open(cfg.format_template_path, "r", encoding="utf-8") as fp:
            project_format_template = json.load(fp)
        file_format_template = FILE_ANALYSIS_TEMPLATE
        if isinstance(project_format_template, dict):
            files_tpl = project_format_template.get("files")
            if isinstance(files_tpl, list) and files_tpl and isinstance(files_tpl[0], dict):
                file_format_template = files_tpl[0]
    else:
        project_format_template = DEFAULT_FORMAT_TEMPLATE
        file_format_template = FILE_ANALYSIS_TEMPLATE

    # Fail fast if API key missing and not a dry-run
    if not cfg.dry_run and not (cfg.api_key and cfg.api_key.strip()):
        raise RuntimeError("Missing API key. Set DEEPSEEK_API_KEY environment variable or pass --api-key.")

    llm = LLMClient(cfg.api_base, cfg.api_key, cfg.model, timeout=cfg.timeout)
    analyzer = ProjectAnalyzer(llm, cfg, project_format_template, file_format_template)

    # Analyze files in parallel
    file_results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
        futures = {}
        for p in files:
            content = read_text_safely(p, cfg.max_file_bytes)
            futures[ex.submit(analyzer.analyze_file, p, content)] = str(p)
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                res = fut.result()
                file_results.append(res)
            except Exception as e:
                logging.error(f"Error analyzing {path}: {e}")

    # Project synthesis
    project_name = root.name
    final = analyzer.analyze_project(file_results, project_name)
    return final

def parse_args() -> AgentConfig:
    parser = argparse.ArgumentParser(description="AI Sub-Agent for Project Analysis (DeepSeek-compatible)")
    parser.add_argument("--root", type=str, required=True, help="Project root directory")
    parser.add_argument("--include", type=str, default="", help="Comma-separated glob patterns to include (default: **/*)")
    parser.add_argument("--exclude", type=str, default=",".join(DEFAULT_EXCLUDES), help="Comma-separated glob patterns to exclude")
    parser.add_argument("--extensions", type=str, default=",".join(DEFAULT_EXTENSIONS), help="Comma-separated file extensions to include")
    parser.add_argument("--max-files", type=int, default=0, help="Max number of files to analyze (0 = no limit)")
    parser.add_argument("--max-file-bytes", type=int, default=1_200_000, help="Max bytes to read per file")
    parser.add_argument("--chunk-size", type=int, default=6000, help="Max characters per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=300, help="Overlap characters between chunks")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of files to analyze in parallel")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="LLM model name (default: deepseek-chat)")
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE, help="LLM API base, e.g., https://api.deepseek.com")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="API key (or set DEEPSEEK_API_KEY env)")
    parser.add_argument("--output", type=str, default="analysis.json", help="Output file path")
    parser.add_argument("--format", type=str, choices=["json", "markdown"], default="json", help="Output format")
    parser.add_argument("--format-template", type=str, default=None, help="Path to a JSON template defining output structure")
    parser.add_argument("--dry-run", action="store_true", help="Do not call LLM, generate placeholders")
    parser.add_argument("--timeout", type=int, default=90, help="HTTP timeout for LLM calls (seconds)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--lan", type=str, choices=["zh", "en"], default=os.getenv("LLM_LANG", "zh"),
                        help="Output language: zh (简体中文) or en (English). Default: zh")
    parser.add_argument("--strict-json", action="store_true",
                        help="Force model to return strict JSON using response_format")
    parser.add_argument("--summary-output", type=str, default=None,
                        help="Write a standalone self-summary markdown to this path")
    parser.add_argument("--no-append-summary", action="store_true",
                        help="Do not append self-summary to the main markdown output")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")

    include = normalize_globs([args.include]) if args.include else []
    exclude = normalize_globs([args.exclude]) if args.exclude else []
    extensions = normalize_globs([args.extensions]) if args.extensions else []

    cfg = AgentConfig(
        root=args.root,
        include=include,
        exclude=exclude,
        extensions=extensions,
        max_files=args.max_files if args.max_files and args.max_files > 0 else None,
        max_file_bytes=args.max_file_bytes,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        concurrency=max(1, args.concurrency),
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
        output=args.output,
        output_format=args.format,
        format_template_path=args.format_template,
        dry_run=args.dry_run,
        timeout=args.timeout,
        lan=args.lan,
        strict_json=args.strict_json,
        summary_output=args.summary_output,
        append_summary=not args.no_append_summary,
    )
    return cfg

def main():
    cfg = parse_args()
    safe_cfg = {**asdict(cfg), "api_key": _mask_secret(cfg.api_key)}
    logging.info(f"Starting analysis with config: {safe_cfg}")

    try:
        result = run_analysis(cfg)

        # Render main output
        rendered = render_output(result, cfg.output_format, lan=cfg.lan)

        # Build deterministic self-summary markdown
        self_summary_md = build_self_summary_md(result, lan=cfg.lan)

        # Write outputs
        Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)

        if cfg.output_format.lower() == "markdown" and cfg.append_summary:
            # Append self-summary to the end of main markdown
            rendered = f"{rendered}\n\n---\n\n{self_summary_md}"

        with open(cfg.output, "w", encoding="utf-8") as f:
            f.write(rendered)
        logging.info(f"Analysis saved to: {cfg.output}")

        # Standalone summary file (optional)
        if cfg.summary_output:
            Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
            with open(cfg.summary_output, "w", encoding="utf-8") as sf:
                # If user wants only the self-summary text in the separate md
                # we still include a header for clarity.
                title = "## 文档自我总结\n\n" if cfg.lan.lower() == "zh" else "## Document Self Summary\n\n"
                sf.write(f"{title}{self_summary_md}")
            logging.info(f"Standalone summary saved to: {cfg.summary_output}")

    except Exception as e:
        logging.exception(f"Failed to complete analysis: {e}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()