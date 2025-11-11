#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Repair Agent (DeepSeek)
--------------------------
功能：
- 自主/半自主进行缺陷检测与修复：读取代码 -> 让 LLM 识别问题 -> 生成“替换型”补丁 -> 应用到本地文件
- 也支持 full_file（整文件替换）与 unified_diff（生成 .patch 供人工审阅）

特点：
- 中文提示词与中文 JSON 输出
- 优先生成“replacement”补丁（包含行区间与新代码），精准替换原错误代码
- 支持 dry-run 预览、交互确认、自动备份 .bak
- 简单 CLI，可针对单文件或目录（按后缀过滤）
- 安全回退：应用失败自动保留原文件与 .bak

准备：
- pip install openai>=1.0.0
- export DEEPSEEK_API_KEY=sk-xxxx
- 可选：设置 DEEPSEEK_MODEL=deepseek-chat

示例：
  # 对单个文件进行检测并半自主修复（逐条确认）
  python ai_repair_agent.py run --path app/services/user_service.py --interactive

  # 直接自主修复（无交互，自动应用）
  python ai_repair_agent.py run --path app --ext .py --apply

  # 仅检测不修改（dry-run），输出结果到 JSON
  python ai_repair_agent.py run --path app --ext .py --dry-run --save report.json

输出补丁 schema（JSON，中文键名）：
{
  "问题列表": [
    {
      "id": "ISSUE-1",
      "文件路径": "path/to/file",
      "行区间": [start, end],        # 1-based，包含两端
      "类型": "安全/正确性/性能/可维护性/其他",
      "严重性": "low/medium/high/critical",
      "描述": "问题概述与影响",
      "建议": "简述修复策略"
    }
  ],
  "补丁": [
    {
      "文件路径": "path/to/file",
      "补丁类型": "replacement | full_file | unified_diff",
      "行区间": [start, end],        # replacement 专用（优先）
      "新代码": "......",             # replacement 专用：用于替换 [start,end] 的完整新片段
      "整文件内容": "......",         # full_file 专用：替换为该完整文件内容
      "内容": "......",               # unified_diff 专用：标准 unified diff
      "说明": "为什么这样改/兼容性/风险"
    }
  ],
  "验证建议": ["单元/集成/安全测试建议..."]
}
"""
from __future__ import annotations

import argparse
import dataclasses
import difflib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# OpenAI 兼容客户端（适配 DeepSeek）
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -----------------------
# 工具与模型调用层
# -----------------------

def strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def ensure_openai_client(api_key: Optional[str], base_url: str) -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("缺少 openai 依赖：请先 pip install openai>=1.0.0")
    key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("需要 DeepSeek API 密钥（设置环境变量 DEEPSEEK_API_KEY）")
    return OpenAI(api_key=key, base_url=base_url.rstrip("/"))


def llm_chat_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    timeout: int = 120,
    max_retries: int = 2,
    enforce_json: bool = True,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response_format = {"type": "json_object"} if enforce_json else None

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
                response_format=response_format,
            )
            content = (resp.choices[0].message.content or "").strip()
            try:
                return json.loads(content)
            except Exception:
                trimmed = strip_code_fences(content)
                try:
                    return json.loads(trimmed)
                except Exception:
                    # 返回原文，便于诊断
                    return {"raw": content}
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(1.5 ** attempt)
            else:
                raise RuntimeError(f"LLM 调用失败（已重试）：{e}") from e
    # 理论上不会到达
    raise RuntimeError(f"LLM 调用失败：{last_err}")


# -----------------------
# Agent 主体
# -----------------------

class AIRepairAgent:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0.2,
        timeout: int = 120,
        max_retries: int = 2,
        enforce_json: bool = True,
        max_lines_per_file: int = 1200,   # 控制单次输入规模，过长会截断（并标注）
        preferred_patch: str = "replacement",  # 优先要求 replacement
    ):
        self.client = ensure_openai_client(api_key, base_url)
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.enforce_json = enforce_json
        self.max_lines_per_file = max_lines_per_file
        self.preferred_patch = preferred_patch

    # ---- Prompt 构建 ----

    def _system_prompt(self) -> str:
        return (
            "你是一名严谨的资深代码修复工程师与应用安全顾问。"
            "任务：在尽量不改变行为/接口的前提下，精准识别缺陷并给出最小化修改的修复方案。"
            "优先策略：参数化与输入校验、安全 API、空值与异常处理、线程/并发安全、资源释放、边界检查、易读与可维护。"
            "输出要求：仅返回符合‘输出契约’的 JSON（中文键名），不要包含 Markdown 代码围栏。"
            "若上下文不足，请明确假设并给出可验证步骤；不要编造不存在的依赖/函数。"
        )

    def _user_prompt_for_file(self, file_path: str, code: str, language_hint: Optional[str]) -> str:
        # 限制文件长度
        lines = code.splitlines()
        truncated = False
        if len(lines) > self.max_lines_per_file:
            code = "\n".join(lines[: self.max_lines_per_file])
            truncated = True

        payload: Dict[str, Any] = {
            "任务": "检测以下源文件中的潜在缺陷，并生成可直接落地的修复补丁（优先 replacement 替换型）。",
            "文件路径": file_path,
            "语言": language_hint or "auto",
            "代码片段（可能截断）": code,
            "注意": ("为保证上下文大小，代码已截断到前 %d 行。" % self.max_lines_per_file) if truncated else "完整文件已提供。",
            "输出契约": {
                "必须包含": ["问题列表", "补丁", "验证建议"],
                "问题项字段": ["id", "文件路径", "行区间", "类型", "严重性", "描述", "建议"],
                "补丁项字段": [
                    "文件路径", "补丁类型", "(replacement: 行区间, 新代码)",
                    "(full_file: 整文件内容)", "(unified_diff: 内容)", "说明"
                ],
                "补丁优先级": f"优先使用 '{self.preferred_patch}'。"
            },
            "补丁规则": [
                "replacement：务必给出准确的 1-based 行区间 [start, end]，‘新代码’需为可直接替换该区间的完整代码片段。",
                "full_file：给出替换后的完整文件内容（不建议，除非改动较大或 replacement 不可行）。",
                "unified_diff：提供标准 unified diff（本 Agent 将保存为 .patch，默认不自动套用）。"
            ],
            "风格与限制": [
                "尽量最小化修改，保持接口与行为一致（除非缺陷本身需要变更）。",
                "不要引入未在上下文出现且无法明确安装的第三方依赖。",
                "必要时在‘说明’中写明兼容性与迁移注意点。"
            ],
            "输出语言": "中文",
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    # ---- 核心流程 ----

    def analyze_file(self, file_path: str, code: str, language_hint: Optional[str] = None) -> Dict[str, Any]:
        return llm_chat_json(
            client=self.client,
            model=self.model,
            system_prompt=self._system_prompt(),
            user_prompt=self._user_prompt_for_file(file_path, code, language_hint),
            temperature=self.temperature,
            timeout=self.timeout,
            max_retries=self.max_retries,
            enforce_json=self.enforce_json,
        )

    # ---- 补丁应用 ----

    @staticmethod
    def _apply_replacement_patch(orig: str, start: int, end: int, new_code: str) -> str:
        """
        将 [start, end]（1-based, inclusive）行替换为 new_code。
        """
        lines = orig.splitlines()
        n = len(lines)
        if start < 1 or end < start or end > n:
            raise ValueError(f"行区间非法：[{start}, {end}] 超出文件总行数 {n}")
        # Python splitlines 不保留换行符，拼接时统一使用 '\n'
        before = lines[: start - 1]
        after = lines[end:]
        new_lines = new_code.splitlines()
        return "\n".join(before + new_lines + after) + ("\n" if orig.endswith("\n") else "")

    @staticmethod
    def _preview_diff(old_text: str, new_text: str, file_path: str) -> str:
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)
        diff = difflib.unified_diff(
            old_lines, new_lines, fromfile=f"a/{file_path}", tofile=f"b/{file_path}", lineterm=""
        )
        return "".join(diff)

    def apply_patches(
        self,
        base_dir: Path,
        result: Dict[str, Any],
        *,
        dry_run: bool = True,
        interactive: bool = False,
        auto_yes: bool = False,
        backup: bool = True,
        save_unified_diff: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        尝试应用 LLM 返回的补丁列表。
        返回每个补丁的应用结果：{"文件路径":..., "补丁类型":..., "状态":"applied|skipped|failed", "原因":..., "预览diff":...}
        """
        patches: List[Dict[str, Any]] = list(result.get("补丁") or [])
        outcomes: List[Dict[str, Any]] = []

        for idx, p in enumerate(patches, 1):
            fpath = p.get("文件路径")
            ptype = p.get("补丁类型")
            outcome = {"文件路径": fpath, "补丁类型": ptype, "状态": "skipped", "原因": "", "预览diff": ""}

            if not fpath or not ptype:
                outcome["状态"] = "failed"
                outcome["原因"] = "补丁缺少必要字段（文件路径/补丁类型）"
                outcomes.append(outcome)
                continue

            abs_path = (base_dir / fpath).resolve()
            if not abs_path.exists():
                outcome["状态"] = "failed"
                outcome["原因"] = f"文件不存在：{abs_path}"
                outcomes.append(outcome)
                continue

            try:
                old = abs_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                old = abs_path.read_text(encoding="utf-8", errors="ignore")

            new_text: Optional[str] = None
            preview: str = ""

            try:
                if ptype == "replacement":
                    line_range = p.get("行区间")
                    new_code = p.get("新代码")
                    if not (isinstance(line_range, list) and len(line_range) == 2 and isinstance(new_code, str)):
                        raise ValueError("replacement 需要提供 [start, end] 与 '新代码'")
                    start, end = int(line_range[0]), int(line_range[1])
                    new_text = self._apply_replacement_patch(old, start, end, new_code)
                    preview = self._preview_diff(old, new_text, fpath)

                elif ptype == "full_file":
                    after_full = p.get("整文件内容")
                    if not isinstance(after_full, str):
                        raise ValueError("full_file 需要提供 '整文件内容'")
                    new_text = after_full
                    preview = self._preview_diff(old, new_text, fpath)

                elif ptype == "unified_diff":
                    diff_text = p.get("内容") or ""
                    outcome["状态"] = "skipped"
                    outcome["原因"] = "unified_diff 默认不自动应用，已保存为 .patch"
                    if save_unified_diff:
                        patch_path = abs_path.with_suffix(abs_path.suffix + f".agent{idx}.patch")
                        patch_path.write_text(diff_text, encoding="utf-8")
                    outcome["预览diff"] = diff_text
                    outcomes.append(outcome)
                    continue

                else:
                    raise ValueError(f"未知补丁类型：{ptype}")

                # 确认阶段
                if dry_run:
                    outcome["状态"] = "skipped"
                    outcome["原因"] = "dry-run 仅预览，不写盘"
                    outcome["预览diff"] = preview
                    outcomes.append(outcome)
                    continue

                if interactive and not auto_yes:
                    print(f"\n[补丁预览 #{idx}] {fpath}\n{preview}\n")
                    ans = input("是否应用该补丁？(y/N): ").strip().lower()
                    if ans not in ("y", "yes"):
                        outcome["状态"] = "skipped"
                        outcome["原因"] = "用户取消"
                        outcome["预览diff"] = preview
                        outcomes.append(outcome)
                        continue

                # 备份
                if backup:
                    bak_path = abs_path.with_suffix(abs_path.suffix + ".bak")
                    bak_path.write_text(old, encoding="utf-8")

                # 写回
                abs_path.write_text(new_text, encoding="utf-8")

                outcome["状态"] = "applied"
                outcome["原因"] = ""
                outcome["预览diff"] = preview
                outcomes.append(outcome)

            except Exception as e:
                outcome["状态"] = "failed"
                outcome["原因"] = f"{type(e).__name__}: {e}"
                outcome["预览diff"] = preview
                outcomes.append(outcome)

        return outcomes


# -----------------------
# CLI
# -----------------------

def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def guess_language_by_suffix(path: Path) -> Optional[str]:
    suf = path.suffix.lower()
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".go": "go",
        ".rb": "ruby",
        ".php": "php",
        ".rs": "rust",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".c": "c",
        ".cs": "csharp",
        ".kt": "kotlin",
        ".swift": "swift",
        ".scala": "scala",
        ".m": "objective-c",
        ".mm": "objective-cpp",
        ".sh": "bash",
    }
    return mapping.get(suf)


def iter_files(base: Path, exts: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in base.rglob("*"):
        if p.is_file():
            if not exts or p.suffix in exts:
                files.append(p)
    return files


def run_agent(
    base_path: Path,
    exts: List[str],
    apply_changes: bool,
    interactive: bool,
    auto_yes: bool,
    dry_run: bool,
    save_json: Optional[Path],
    model: str,
    temperature: float,
    enforce_json: bool,
    max_lines: int,
) -> int:
    agent = AIRepairAgent(
        model=model,
        temperature=temperature,
        enforce_json=enforce_json,
        max_lines_per_file=max_lines,
    )

    targets: List[Path] = []
    if base_path.is_file():
        targets = [base_path]
    else:
        targets = iter_files(base_path, exts)

    all_results: Dict[str, Any] = {"文件清单": [], "结果": []}
    base_dir = base_path if base_path.is_dir() else base_path.parent

    for path in targets:
        rel = str(path.relative_to(base_dir))
        code = load_text(path)
        lang = guess_language_by_suffix(path)

        print(f"分析：{rel} ...")
        result = agent.analyze_file(rel, code, lang)
        all_results["文件清单"].append(rel)
        all_results["结果"].append({"文件": rel, "LLM输出": result})

        # 应用补丁
        outcomes = agent.apply_patches(
            base_dir=base_dir,
            result=result if isinstance(result, dict) else {},
            dry_run=dry_run,
            interactive=interactive,
            auto_yes=auto_yes,
        )
        # 附加到结果
        all_results["结果"][-1]["应用结果"] = outcomes

        # 打印简要状态
        for oc in outcomes:
            status = oc.get("状态")
            reason = oc.get("原因") or ""
            print(f"  - [{status}] {oc.get('文件路径')}: {reason}")

    if save_json:
        save_json.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已保存结果到：{save_json}")

    # 返回码：存在 failed 则非零
    any_failed = any(
        any(oc.get("状态") == "failed" for oc in (item.get("应用结果") or []))
        for item in all_results["结果"]
    )
    return 1 if any_failed else 0


def main():
    parser = argparse.ArgumentParser(
        description="缺陷检测与修复 AI Agent（DeepSeek） - 生成替换型新代码并应用到文件"
    )
    parser.add_argument(
        "run",
        nargs="?",
        default="run",
        help="命令（固定为 run）",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="文件或目录路径",
    )
    parser.add_argument(
        "--ext",
        action="append",
        default=[],
        help="当 path 为目录时，按后缀过滤（可多次指定，如 --ext .py --ext .js）。留空表示全部文件",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="自主模式：直接应用补丁（非 dry-run）",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="半自主模式：应用前逐条确认",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="配合 --interactive 使用，跳过确认直接应用",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览不写盘（默认 false）。若未指定 --apply，等价于开启 dry-run。",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="将完整 JSON 结果保存到文件（可选）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        help="DeepSeek 模型名（默认：deepseek-chat）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("DEEPSEEK_TEMPERATURE", "0.2")),
        help="采样温度（默认 0.2）",
    )
    parser.add_argument(
        "--no-enforce-json",
        action="store_true",
        help="不强制 LLM 使用 JSON 响应（兼容某些代理）",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=int(os.getenv("AGENT_MAX_LINES", "1200")),
        help="单文件最大分析行数（超出将截断）",
    )

    args = parser.parse_args()
    base_path = Path(args.path).resolve()
    if not base_path.exists():
        print(f"路径不存在：{base_path}", file=sys.stderr)
        sys.exit(2)

    # dry-run 默认：如果未显式 --apply，则启用 dry-run
    dry_run = args.dry_run or (not args.apply)
    save_json = Path(args.save).resolve() if args.save else None

    code = run_agent(
        base_path=base_path,
        exts=args.ext,
        apply_changes=args.apply,
        interactive=args.interactive,
        auto_yes=args.yes,
        dry_run=dry_run,
        save_json=save_json,
        model=args.model,
        temperature=args.temperature,
        enforce_json=not args.no_enforce_json,
        max_lines=args.max_lines,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()