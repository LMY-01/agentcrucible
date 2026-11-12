"""内置工具集：全部限制在 workspace 目录内。

安全边界靠 pathlib 解析后的 commonpath 判断，不靠字符串前缀匹配——
`../../etc/passwd` 这种能绕过前缀检查，绕过不了 realpath。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .registry import NET, READ, WRITE, Tool

MAX_OUTPUT_CHARS = 4000


def _safe_join(root: Path, rel: str) -> Path:
    target = (root / rel).resolve()
    root_r = root.resolve()
    if root_r != target and root_r not in target.parents:
        raise PermissionError(f"路径 {rel!r} 越出了工作区 {root_r}")
    return target


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n... [省略 {len(text) - limit} 字符] ...\n{tail}"


def build_tools(workspace: str, skill_loader=None) -> list[Tool]:
    root = Path(workspace).resolve()

    def read_file(path: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
        target = _safe_join(root, path)
        if not target.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        if target.is_dir():
            raise IsADirectoryError(f"{path} 是目录，用 list_dir")
        return _truncate(target.read_text(encoding="utf-8", errors="replace"), max_chars)

    def write_file(path: str, content: str) -> str:
        target = _safe_join(root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"已写入 {path}（{len(content)} 字符）"

    def list_dir(path: str = ".") -> str:
        target = _safe_join(root, path)
        if not target.is_dir():
            raise NotADirectoryError(f"{path} 不是目录")
        rows = []
        for item in sorted(target.iterdir()):
            suffix = "/" if item.is_dir() else f"  {item.stat().st_size}B"
            rows.append(f"{item.name}{suffix}")
        return "\n".join(rows) or "(空目录)"

    def run_python(code: str, timeout: int = 20) -> str:
        timeout = max(1, min(int(timeout), 60))
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        return _truncate(f"exit={proc.returncode}\n{out}")

    tools = [
        Tool(
            name="read_file",
            description="读取工作区内的文本文件。大文件返回头尾，中间省略。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对工作区的路径"},
                    "max_chars": {"type": "integer", "description": "最多返回多少字符"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=read_file,
            permission=READ,
        ),
        Tool(
            name="write_file",
            description="把内容写入工作区内的文件，目录不存在会自动创建。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            handler=write_file,
            permission=WRITE,
        ),
        Tool(
            name="list_dir",
            description="列出工作区某个目录下的文件名和大小。",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": [],
                "additionalProperties": False,
            },
            handler=list_dir,
            permission=READ,
        ),
        Tool(
            name="run_python",
            description="在工作区目录下执行一段 Python 代码，返回 stdout/stderr 和 exit code。用于计算、验证、跑脚本。",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "要执行的 Python 源码"},
                    "timeout": {"type": "integer", "description": "超时秒数，1-60"},
                },
                "required": ["code"],
                "additionalProperties": False,
            },
            handler=run_python,
            permission=NET,  # 执行代码等价于完全放权，归到最高权限档
        ),
    ]

    if skill_loader is not None:
        tools.append(
            Tool(
                name="load_skill",
                description="按名字加载一个技能手册的完整内容。只有当技能目录里的简述不足以完成任务时才调用。",
                parameters={
                    "type": "object",
                    "properties": {"name": {"type": "string", "description": "技能名"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
                handler=lambda name: skill_loader.load_body(name),
                permission=READ,
            )
        )
    return tools
