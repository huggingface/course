# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import re
import unittest
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

_CONTINUATION_LABEL_PATTERN = re.compile(r"^(?P<base>.+):(?P<suffix>[1-9]\d*)$")
_PYTEST_DECORATOR_PATTERN = re.compile(r"^#\s*pytest-decorator:\s*(.+)$", re.MULTILINE)


def _resolve_decorator(dotted_path: str):
    """Import and return a decorator from a dotted path like 'transformers.testing_utils.slow'."""
    module_path, _, attr_name = dotted_path.rpartition(".")
    if not module_path:
        raise ImportError(f"Cannot resolve decorator: {dotted_path!r} (no module path)")
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


@dataclass
class RunnableBlock:
    code: str
    name: str
    idx: int
    decorators: list[str] = field(default_factory=list)


class DocIntegrationTest(unittest.TestCase):
    """Base class to execute runnable Python code blocks embedded in markdown docs."""

    __test__ = False  # Prevent pytest from collecting the base class itself.

    # Path to the markdown file; subclasses should override.
    doc_path: Path | None = None
    # Optional cleanup hook called after each block.
    cleanup_func: Callable[[], None] | None = None
    # Flag that marks fenced code blocks that should be executed.
    runnable_flag = "runnable"
    # Internal switch: when True, per-block test methods are generated.
    _dynamic_tests_created = False

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        blocks = cls._collect_runnable_blocks()
        if not blocks:
            # Allow pytest to collect subclasses that provide their own doc_path without blocks.
            cls.__test__ = True
            return

        cls._dynamic_tests_created = True
        cls.__test__ = True  # Enable collection for subclasses with runnable blocks.

        for block in blocks:
            method_name = block.name if block.name.startswith("test") else f"test_{block.name}"
            if hasattr(cls, method_name):
                # Don't override existing explicit tests.
                continue

            def _make_test(code: str, name: str, block_method_name: str, decorator_paths: list[str]):
                def _test(self):
                    self._execute_block(code, name)

                _test.__name__ = block_method_name
                for path in decorator_paths:
                    try:
                        decorator = _resolve_decorator(path)
                        _test = decorator(_test)
                    except (ImportError, AttributeError):
                        pass
                return _test

            setattr(cls, method_name, _make_test(block.code, block.name, method_name, block.decorators))

        # Avoid double-reporting: neutralize the catch-all method when per-block tests exist.
        if hasattr(cls, "test_markdown_runnable_blocks"):
            cls.test_markdown_runnable_blocks = None

    @classmethod
    def _collect_runnable_blocks(cls) -> list[RunnableBlock]:
        if cls.doc_path is None:
            return []

        resolved_path = Path(cls.doc_path)
        if not resolved_path.exists():
            return []

        doc_text = resolved_path.read_text(encoding="utf-8")
        return cls._collect_runnable_blocks_from_text(doc_text)

    @classmethod
    def _collect_runnable_blocks_from_text(cls, text: str) -> list[RunnableBlock]:
        blocks = []
        block_indices_by_name = {}
        for idx, (code, name, decorators) in enumerate(cls._iter_runnable_code_blocks(text)):
            safe_name = name or f"doc_block_{idx}"
            continuation_match = _CONTINUATION_LABEL_PATTERN.match(safe_name)
            if continuation_match:
                base_name = continuation_match.group("base")
                base_block_idx = block_indices_by_name.get(base_name)
                if base_block_idx is not None:
                    blocks[base_block_idx].code = f"{blocks[base_block_idx].code}\n\n{code}"
                    continue

            block_indices_by_name[safe_name] = len(blocks)
            blocks.append(RunnableBlock(code=code, name=safe_name, idx=idx, decorators=decorators))

        return blocks

    @classmethod
    def _iter_runnable_code_blocks(cls, text: str) -> Iterable[tuple[str, str | None, list[str]]]:
        """
        Yield (code, name, decorators) tuples for blocks marked with the runnable flag.
        Supports optional naming via a suffix: ```py runnable:test_name
        A later block named ```py runnable:test_name:2``` continues the earlier test.

        Code blocks may contain ``# pytest-decorator: dotted.path.to.decorator`` comment lines.
        These are extracted as decorator paths and stripped from the code before execution.
        Multiple decorators can be comma-separated on a single line.
        """
        pattern = re.compile(r"```(?P<lang>python|py)(?P<flags>[^\n]*)\n(?P<code>.+?)\n```", re.DOTALL)
        for match in pattern.finditer(text):
            flags = [flag.strip() for flag in match.group("flags").split() if flag.strip()]
            runnable_flags = [
                flag for flag in flags if flag == cls.runnable_flag or flag.startswith(f"{cls.runnable_flag}:")
            ]
            if not runnable_flags:
                continue
            name = None
            for flag in runnable_flags:
                if ":" in flag:
                    _, name = flag.split(":", 1)

            code = match.group("code")
            decorators = []
            for dec_match in _PYTEST_DECORATOR_PATTERN.finditer(code):
                for path in dec_match.group(1).split(","):
                    path = path.strip()
                    if path:
                        decorators.append(path)
            code = _PYTEST_DECORATOR_PATTERN.sub("", code).strip()

            yield code, name, decorators

    def _run_cleanup(self):
        if callable(self.cleanup_func):
            self.cleanup_func()

    def _execute_block(self, code: str, name: str):
        resolved_path = Path(self.doc_path) if self.doc_path is not None else Path("unknown_doc.md")
        namespace = {"__name__": f"{resolved_path.stem}_{name}"}
        try:
            exec(compile(code, str(resolved_path), "exec"), namespace)
        except Exception as err:
            raise AssertionError(f"Doc block '{name}' in {resolved_path} failed to execute:\n{code}") from err
        finally:
            self._run_cleanup()

    def test_markdown_runnable_blocks(self):
        # If dynamic per-block tests are generated, avoid double-running.
        if self._dynamic_tests_created:
            self.skipTest("Per-block doc tests generated dynamically; run those instead.")

        if self.doc_path is None:
            self.skipTest("doc_path not set on DocIntegrationTest subclass")

        resolved_path = Path(self.doc_path)
        runnable_blocks = self._collect_runnable_blocks()

        self.assertTrue(runnable_blocks, f"No runnable code blocks were found in {resolved_path}")

        for block in runnable_blocks:
            with self.subTest(block=block.name):
                self._execute_block(block.code, block.name)
