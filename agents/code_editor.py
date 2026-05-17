"""
CodeEditor: Thin code editing layer that makes a single LLM API call to modify target files.

Replaces Aider for fairer benchmarking — no hidden intelligence, full prompt logged
for transparency.
"""

import ast
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import backoff
import openai

from .llm import create_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are modifying existing Python code to improve machine learning performance.

Rules:
- You will receive target files you can modify and a task description for context.
- Make targeted, minimal changes that address the instruction.
- Preserve imports, class structure, and function signatures unless the change specifically requires modifying them.
- For small files: return the COMPLETE modified file content.
- For large files: return only SEARCH/REPLACE blocks for the changed sections."""

MAX_COMPLETION_TOKENS = 16384

# Patterns that indicate dangerous filesystem operations in LLM-generated code.
# ML algorithm files should never need these — block before writing to disk.
DANGEROUS_CODE_PATTERNS = [
    # File deletion
    (r'\bos\s*\.\s*remove\s*\(', "os.remove()"),
    (r'\bos\s*\.\s*unlink\s*\(', "os.unlink()"),
    (r'\bos\s*\.\s*rmdir\s*\(', "os.rmdir()"),
    (r'\bos\s*\.\s*removedirs\s*\(', "os.removedirs()"),
    # File move/rename
    (r'\bos\s*\.\s*rename\s*\(', "os.rename()"),
    (r'\bos\s*\.\s*replace\s*\(', "os.replace()"),
    # Recursive delete/copy via shutil
    (r'\bshutil\s*\.\s*rmtree\s*\(', "shutil.rmtree()"),
    (r'\bshutil\s*\.\s*move\s*\(', "shutil.move()"),
    (r'\bshutil\s*\.\s*copy\w*\s*\(', "shutil.copy*()"),
    # Shell execution
    (r'\bos\s*\.\s*system\s*\(', "os.system()"),
    (r'\bos\s*\.\s*popen\s*\(', "os.popen()"),
    (r'\bos\s*\.\s*exec\w*\s*\(', "os.exec*()"),
    (r'\bos\s*\.\s*spawn\w*\s*\(', "os.spawn*()"),
    # Subprocess (any subprocess call is a shell escape vector)
    (r'\bsubprocess\s*\.\s*\w+\s*\(', "subprocess.*()"),
    # Dynamic imports
    (r'\b__import__\s*\(', "__import__()"),
    # Pathlib destructive
    (r'\.\s*unlink\s*\(', ".unlink()"),
    (r'\.\s*rmdir\s*\(', ".rmdir()"),
]


@dataclass
class EditResult:
    """Result of a single code editing LLM call."""

    success: bool
    files_changed: List[str] = field(default_factory=list)
    error: Optional[str] = None
    token_usage: Optional[dict] = None
    raw_response: Optional[str] = None
    log_path: Optional[str] = None  # Path to the prompt+response log file


class CodeEditor:
    """
    Thin editing layer: instruction -> LLM call -> write target_files.
    No hidden intelligence. Full prompt logged for transparency.
    """

    FILE_SIZE_THRESHOLD = 500  # lines; above this, use search/replace format

    def __init__(
        self,
        model: str,
        provider: str,
        target_files: List[str],
        task_description: str,
        log_dir: Optional[str] = None,
        metric_name: Optional[str] = None,
        metric_direction: Optional[str] = None,
    ):
        """
        Args:
            model: LLM model name (e.g., "gpt-5-2025-08-07").
            provider: LLM provider (e.g., "OpenAI", "OpenRouter", "Google").
            target_files: Absolute paths to files the agent can edit.
            task_description: Task description from prompt.json.
            log_dir: Directory for saving prompt/response logs. If None, logging
                     to file is skipped.
            metric_name: Name of the primary metric (e.g., "accuracy_mean").
            metric_direction: Whether higher or lower is better ("higher" / "lower").
        """
        self.client, self.model_name = create_client(model, provider)
        self.target_files = [os.path.abspath(f) for f in target_files]
        self.task_description = task_description
        self.log_dir = log_dir
        self.metric_name = metric_name
        self.metric_direction = metric_direction
        self._call_count = 0

        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @backoff.on_exception(
        backoff.expo, (openai.RateLimitError, openai.APITimeoutError), max_tries=5
    )
    def edit(self, instruction: str, error_context: Optional[str] = None) -> EditResult:
        """
        Make ONE LLM call to edit the target files according to *instruction*.

        Args:
            instruction: Natural-language editing instruction for the LLM.
            error_context: Optional stderr / traceback from a previous failed run.

        Returns:
            EditResult with success status, changed files, and token usage.
        """
        # 1. Read current content of all target files
        try:
            targets = self._read_targets()
        except Exception as e:
            return EditResult(success=False, error=f"Failed to read target files: {e}")

        # 2. Build prompt
        system_prompt, user_prompt = self._build_prompt(instruction, targets, error_context)

        # 3. Single LLM call
        try:
            # Build API kwargs — use defaults only (many models reject
            # custom temperature or max_tokens)
            api_kwargs = dict(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            response = self.client.chat.completions.create(**api_kwargs)
            content = response.choices[0].message.content
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return EditResult(success=False, error=f"LLM call failed: {e}")

        # 4. Log prompt + response
        log_path = self._log(user_prompt, content)

        # 5. Parse response and apply changes
        try:
            modified = self._parse_and_apply(content, targets)
        except Exception as e:
            logger.error("Failed to parse/apply LLM response: %s", e)
            return EditResult(
                success=False,
                error=f"Failed to parse/apply response: {e}",
                token_usage=token_usage,
                raw_response=content,
                log_path=log_path,
            )

        return EditResult(
            success=bool(modified),
            files_changed=modified,
            token_usage=token_usage,
            raw_response=content,
            log_path=log_path,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        instruction: str,
        targets: Dict[str, str],
        error_context: Optional[str],
    ) -> Tuple[str, str]:
        """Build system and user prompts for the LLM call.

        Returns:
            (system_prompt, user_prompt)
        """
        parts: List[str] = []

        # Task context
        parts.append(f"## Task Context\n{self.task_description}")

        # Optimization target (U4: metric name + direction)
        if self.metric_name:
            dir_str = f" ({self.metric_direction} is better)" if self.metric_direction else ""
            parts.append(f"## Optimization Target\nMetric: {self.metric_name}{dir_str}")

        # Classify files by size
        small = {f: c for f, c in targets.items() if c.count("\n") <= self.FILE_SIZE_THRESHOLD}
        large = {f: c for f, c in targets.items() if c.count("\n") > self.FILE_SIZE_THRESHOLD}

        # Small files — show full, ask for complete replacement
        if small:
            parts.append("## Target files (return COMPLETE modified content):")
            for path, content in small.items():
                parts.append(f"### FILE: {path}\n```python\n{content}\n```")

        # Large files — show full, ask for search/replace blocks
        if large:
            parts.append("## Large target files (return SEARCH/REPLACE blocks):")
            for path, content in large.items():
                parts.append(f"### FILE: {path}\n```python\n{content}\n```")

        # Error context from previous execution
        if error_context:
            parts.append(f"## Previous execution error:\n```\n{error_context}\n```")

        # The actual instruction
        parts.append(f"## Instruction:\n{instruction}")

        # Implementation guidelines (U3: unified across all agents)
        parts.append(
            "## Implementation guidelines\n"
            "- Write complete, self-contained code. Do not skip sections or leave placeholders.\n"
            "- Do not add additional command line arguments.\n"
            "- Ensure the code runs without errors.\n"
            "- Do not modify files other than the target files shown above."
        )

        # Output format specification
        parts.append(self._output_format_instructions(has_small=bool(small), has_large=bool(large)))

        user_prompt = "\n\n".join(parts)
        return SYSTEM_PROMPT, user_prompt

    @staticmethod
    def _output_format_instructions(has_small: bool, has_large: bool) -> str:
        """Generate output format instructions based on file sizes."""
        sections: List[str] = ["## Output Format\n"]

        if has_small:
            sections.append(
                "For small target files, return the COMPLETE modified file using this format:\n"
                "### FILE: <filepath>\n"
                "```python\n"
                "<complete file content>\n"
                "```"
            )

        if has_large:
            sections.append(
                "For large target files, return only the changed sections using SEARCH/REPLACE blocks:\n"
                "### FILE: <filepath>\n"
                "<<<<<<< SEARCH\n"
                "<exact existing code to find>\n"
                "=======\n"
                "<replacement code>\n"
                ">>>>>>> REPLACE\n"
                "\n"
                "You may include multiple SEARCH/REPLACE blocks per file."
            )

        sections.append(
            "Important:\n"
            "- Use the exact file paths shown above.\n"
            "- Ensure all code is syntactically valid Python.\n"
            "- Do not add explanations outside the specified format."
        )

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Response parsing and application
    # ------------------------------------------------------------------

    def _parse_and_apply(self, response: str, original_targets: Dict[str, str]) -> List[str]:
        """Parse LLM response and apply edits to target files.

        Handles two formats:
          1. Whole-file replacement for small files.
          2. Search/replace blocks for large files.

        Returns:
            List of absolute paths of files that were successfully modified.
        """
        modified: List[str] = []

        # Whole-file outputs (small files)
        for filepath, code in self._extract_whole_files(response):
            resolved = self._resolve_path(filepath, original_targets)
            if resolved is None:
                logger.warning("Skipping unrecognized file path from LLM: %s", filepath)
                continue
            if not self._syntax_ok(code):
                logger.warning("Syntax error in whole-file output for %s — skipping.", resolved)
                continue
            is_safe, violations = self._check_code_safety(code)
            if not is_safe:
                logger.warning("BLOCKED: Dangerous code in %s: %s", resolved, violations)
                continue
            Path(resolved).write_text(code)
            if resolved not in modified:
                modified.append(resolved)

        # Search/replace blocks (large files)
        for filepath, search, replace in self._extract_search_replace(response):
            resolved = self._resolve_path(filepath, original_targets)
            if resolved is None:
                logger.warning("Skipping unrecognized file path from LLM: %s", filepath)
                continue
            content = Path(resolved).read_text()
            if search not in content:
                logger.warning(
                    "SEARCH block not found in %s — skipping this replacement.", resolved
                )
                continue
            new_content = content.replace(search, replace, 1)
            if not self._syntax_ok(new_content):
                logger.warning(
                    "Syntax error after applying SEARCH/REPLACE to %s — skipping.", resolved
                )
                continue
            is_safe, violations = self._check_code_safety(replace)
            if not is_safe:
                logger.warning(
                    "BLOCKED: Dangerous code in REPLACE block for %s: %s", resolved, violations
                )
                continue
            Path(resolved).write_text(new_content)
            if resolved not in modified:
                modified.append(resolved)

        return modified

    @staticmethod
    def _extract_whole_files(response: str) -> List[Tuple[str, str]]:
        """Extract whole-file blocks from the LLM response.

        Expected format:
            ### FILE: <path>
            ```python
            <code>
            ```

        Returns:
            List of (filepath, code) tuples.
        """
        pattern = r"### FILE:\s*(\S+)\s*\n```(?:python)?\s*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        # Strip trailing whitespace/newlines from extracted code
        return [(filepath, code.rstrip("\n")) for filepath, code in matches]

    @staticmethod
    def _extract_search_replace(response: str) -> List[Tuple[str, str, str]]:
        """Extract search/replace blocks from the LLM response.

        Expected format:
            ### FILE: <path>
            <<<<<<< SEARCH
            <search_text>
            =======
            <replace_text>
            >>>>>>> REPLACE

        Returns:
            List of (filepath, search_text, replace_text) tuples.
        """
        pattern = (
            r"### FILE:\s*(\S+)\s*\n"
            r"<<<<<<< SEARCH\n"
            r"(.*?)\n"
            r"=======\n"
            r"(.*?)\n"
            r">>>>>>> REPLACE"
        )
        return re.findall(pattern, response, re.DOTALL)

    @staticmethod
    def _syntax_ok(code: str) -> bool:
        """Check whether *code* is syntactically valid Python.

        Returns True if ast.parse succeeds, False otherwise.
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    @staticmethod
    def _check_code_safety(code: str) -> Tuple[bool, List[str]]:
        """Check code for dangerous filesystem operations.

        Uses regex for most patterns and AST walk for exec/eval
        (to avoid false positives like model.eval()).

        Returns:
            (is_safe, list_of_violations)
        """
        violations = []

        # Regex-based checks
        for pattern, name in DANGEROUS_CODE_PATTERNS:
            if re.search(pattern, code):
                violations.append(name)

        # AST-based checks for exec/eval (avoids false positives like model.eval())
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in ("exec", "eval"):
                        violations.append(f"{node.func.id}() [line {node.lineno}]")
        except SyntaxError:
            pass  # Syntax check is done separately

        return (len(violations) == 0, violations)

    @staticmethod
    def _resolve_path(filepath: str, original_targets: Dict[str, str]) -> Optional[str]:
        """Match a filepath from the LLM response to one of the original target paths.

        Supports:
          - Exact match (full absolute path).
          - Basename match (e.g., LLM returns "train.py" and we have "/abs/path/train.py").
          - Suffix match (e.g., LLM returns "src/train.py" matching "/workspace/src/train.py").

        Returns:
            The resolved absolute path, or None if no match is found.
        """
        # Exact match
        if filepath in original_targets:
            return filepath

        # Normalize for comparison
        filepath_clean = filepath.strip().rstrip("/")

        for target_path in original_targets:
            # Basename match
            if os.path.basename(target_path) == os.path.basename(filepath_clean):
                return target_path
            # Suffix match
            if target_path.endswith(filepath_clean):
                return target_path

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_targets(self) -> Dict[str, str]:
        """Read the current content of all target files.

        Returns:
            Dict mapping absolute file path to file content.

        Raises:
            FileNotFoundError: If a target file does not exist.
        """
        targets: Dict[str, str] = {}
        for filepath in self.target_files:
            p = Path(filepath)
            if not p.exists():
                raise FileNotFoundError(f"Target file not found: {filepath}")
            targets[filepath] = p.read_text()
        return targets

    def _log(self, prompt: str, response: str) -> Optional[str]:
        """Save the full prompt and response to a log file for transparency.

        Returns the log file path on success, or None.
        """
        self._call_count += 1

        if not self.log_dir:
            logger.debug(
                "CodeEditor call #%d completed (no log_dir set, skipping file log).",
                self._call_count,
            )
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(
            self.log_dir, f"code_editor_{timestamp}_call{self._call_count}.txt"
        )

        with open(log_file, "w") as f:
            f.write("=" * 60 + "\n")
            f.write(f"CodeEditor Call #{self._call_count}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write("--- PROMPT ---\n")
            f.write(prompt)
            f.write("\n\n--- RESPONSE ---\n")
            f.write(response if response else "<empty response>")
            f.write("\n")

        logger.info("CodeEditor call #%d logged to %s", self._call_count, log_file)
        return log_file
