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

"""Pytest plugin that collects runnable code blocks from markdown files."""

import importlib
import re
import tempfile
from pathlib import Path

import pytest

from doc_builder.testing import DocIntegrationTest

# Matches lines like: # pytest-decorator: transformers.testing_utils.slow
_DECORATOR_RE = re.compile(r"^#\s*pytest-decorator:\s*(.+)$")


def _import_decorator(dotted_path):
    """Import a decorator from a dotted path like ``package.module.name``."""
    module_path, _, attr = dotted_path.rstrip().rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _parse_decorators(code):
    """Extract ``# pytest-decorator: ...`` lines from code.

    Returns (cleaned_code, list_of_decorator_callables).
    """
    lines = code.split("\n")
    decorators = []
    clean_lines = []
    for line in lines:
        m = _DECORATOR_RE.match(line.strip())
        if m:
            for dotted_path in m.group(1).split(","):
                dotted_path = dotted_path.strip()
                if dotted_path:
                    decorators.append(_import_decorator(dotted_path))
        else:
            clean_lines.append(line)
    return "\n".join(clean_lines), decorators


def pytest_collect_file(parent, file_path):
    if file_path.suffix == ".md":
        return MarkdownFile.from_parent(parent, path=file_path)


class MarkdownFile(pytest.File):
    def collect(self):
        text = self.path.read_text(encoding="utf-8")
        blocks = DocIntegrationTest._collect_runnable_blocks_from_text(text)
        for block in blocks:
            code, decorators = _parse_decorators(block.code)
            yield MarkdownCodeItem.from_parent(self, name=block.name, code=code, decorators=decorators)


class MarkdownCodeItem(pytest.Item):
    def __init__(self, name, parent, code, decorators=None):
        super().__init__(name, parent)
        self._code = code
        self._decorators = decorators or []

    def runtest(self):
        namespace = {"__name__": f"{self.path.stem}_{self.name}"}
        # Write code to a temp .py file so pdb can display the correct source lines.
        self._tmp_file = Path(tempfile.gettempdir()) / f"_doctest_{self.path.stem}_{self.name}.py"
        self._tmp_file.write_text(self._code, encoding="utf-8")

        def execute():
            exec(compile(self._code, str(self._tmp_file), "exec"), namespace)

        # Apply decorators (outermost first → apply in reverse so innermost wraps first)
        for decorator in reversed(self._decorators):
            execute = decorator(execute)

        execute()

    def teardown(self):
        if hasattr(self, "_tmp_file"):
            self._tmp_file.unlink(missing_ok=True)

    def repr_failure(self, excinfo):
        return f"Runnable block '{self.name}' in {self.path} failed:\n\n{excinfo.getrepr()}\n\nCode:\n{self._code}"

    def reportinfo(self):
        return self.path, 0, f"runnable:{self.name}"
