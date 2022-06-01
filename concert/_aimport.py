"""Internal module for the support of imports with top-level `await' (outside of `async def'
functions).
"""
import __future__
import ast
import asyncio
import importlib
import inspect
import logging
import os
import re
import sys
import threading


LOG = logging.getLogger(__name__)
_ASYNC_IMPORT_PATH = []


def register(paths):
    """Register entries in the list of *paths* for importing with enabled top-level await."""
    global _ASYNC_IMPORT_PATH

    if not _ASYNC_IMPORT_PATH:
        _ASYNC_IMPORT_PATH = paths
        sys.meta_path.insert(0, _AsyncMetaPathFinder())


def eval_source(source: bytes, module_dict: dict, filename: str = ''):
    """
    Source code *source* (a byte array) is split into import and non-import statements. All import
    statements are executed one-by-one and all other statements are aggregated and executed
    together. This way we allow nesting of module imports which have top-level `await' keywords.
    This is possible thanks to the fact that when we call :func:`eval` on an import statement, it
    will not return a coroutine but it will invoke the import mechanism (thus us) on the module to
    be imported.  Then that nested module is processed with the loop run if necessary and stopped.
    Then the recursion ends and the importing module processes the rest of the statements with
    possible loop runs which is fine becuase the nested module is processed already, so no nested
    loop runs are attempted.
    """
    def _is_import(node):
        return isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom)

    def _eval_submodule(start, stop):
        # Make an empty tree and fill it with a portion of *nodes* from the *module* tree
        submodule = ast.parse('')
        submodule.body = nodes[start:stop]
        cobj = compile(
            submodule,
            f'{filename or "<string>"}',
            'exec',
            dont_inherit=True,
            flags=compiler_flags
        )

        result = eval(cobj, module_dict)

        if inspect.iscoroutine(result):
            try:
                loop = asyncio.get_event_loop_policy().get_event_loop()
            except RuntimeError as err:
                if threading.current_thread() is not threading.main_thread():
                    # If we are in a different thread we create the loop if not existing
                    # (RuntimeError raised).
                    loop = asyncio.get_event_loop_policy().new_event_loop()
                else:
                    raise err

            try:
                LOG.debug(
                    'import running in loop for %s, nodes %d-%d',
                    filename or '<string>',
                    start,
                    stop
                )
                loop.run_until_complete(result)
            except RuntimeError as err:
                # Actually, if we ever need this we can split the execution even on a finer level
                # and allow mixing `await' and `run_in_loop' (by executing those lines separately)
                name = os.path.basename(os.path.splitext(filename)[0])
                raise ImportError(
                    f"Error loading module `{name}'.\n"
                    # 1
                    f"Possible cause 1: `{name}' uses top-level `await' (i.e. outside of `async "
                    "def' functions) but at the same time tries to execute coroutines in a loop "
                    "at the top level. "
                    "To fix this, remove the loop-running code from the top level of the module "
                    "which has a top-level `await'.\n"
                    # 2
                    f"Possible cause 2: `{name}' was tried to be imported from another module "
                    "from within a running loop, e.g. from an `async def' function. "
                    "To fix this, put all imports to the top level (outside of functions).",
                    path=filename,
                    name=name
                ) from err

    compiler_flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
    # *nodes* are all ast nodes of the module
    nodes = ast.parse(source).body

    if not nodes:
        # Nothing to compile
        return

    # Add compiler flags based on __future__ imports
    comment_pattern = b'#.*\n'
    # Strip comments at the beginning
    stripped_source = re.sub(comment_pattern, b'', source)
    # Look for docstring
    docstring_pattern = b'(r|u)?"""'
    docstring_pattern_obj = re.compile(docstring_pattern)
    # 1 if the first expression is dosctring, 0 otherwise
    first_future_search = int(docstring_pattern_obj.match(stripped_source) is not None)
    future_statements_done = False

    for node in nodes[first_future_search:]:
        if isinstance(node, ast.ImportFrom):
            if node.module == '__future__':
                if future_statements_done:
                    try:
                        source_dec = source.decode('ascii')
                        line_source = ': `' + ast.get_source_segment(source_dec, node) + "'\n"
                    except NameError:
                        # Only in Python 3.8+
                        line_source = ''
                    raise SyntaxError(
                        "`from __future__' imports must occur at the beginning of the file, but "
                        f"file \"{filename}\", contains one on line {node.lineno}{line_source}"
                    )
                for name in node.names:
                    compiler_flags |= getattr(__future__, name.name).compiler_flag
        else:
            future_statements_done = True

    # Do not execute statement-by-statement but aggregate non-import statements for performance
    start = stop = -1
    for i, node in enumerate(nodes):
        if _is_import(node):
            if stop > start:
                # There were non-import statements before us
                _eval_submodule(start, stop)
            # Evaluate us (the import statement) and reset indices
            _eval_submodule(i, i + 1)
            start = stop = -1
        else:
            if start == -1:
                # Initialize *start* if we just started or were on an import statement
                start = i
            # count the lines which can be aggregated (non-import) (open interval)
            stop = i + 1

    if stop > start:
        # Leftover code after last import statement
        _eval_submodule(start, stop)


class _AsyncLoader(importlib.machinery.SourceFileLoader):
    """
    Normal SourceFileLoader with the capability of dealing with `await' outside of `async def'
    functions. The trick is the flag ast.PyCF_ALLOW_TOP_LEVEL_AWAIT and eval() instead of exec()
    used by Python's :class:`_LoaderBasics` class.

    If the module has `await' outside of a function, eval() will return a coroutine which we execute
    in the session's loop, that is why sys.meta_path must be updated *after* the loop has been
    created (to make sure it's the one IPython also uses).
    """
    def exec_module(self, module):
        """
        Execute *module* step-wise and allow *await* on the module level and module nesting with
        top-level `await'.
        """
        eval_source(self.get_data(self.path), module.__dict__, filename=self.path)


class _AsyncMetaPathFinder(importlib.abc.MetaPathFinder):
    """
    MetaPathFinder implementation which used :class;`.FileFinder` for dealing with paths and
    instantiated with our _AsyncLoader for the handling of `await' outside of `async def' functions.
    """
    def find_spec(self, fullname, path, target=None):
        global _ASYNC_IMPORT_PATH

        if path is None:
            # If this is a top-level import, *path* is None and we look in the registered paths
            path = _ASYNC_IMPORT_PATH
        else:
            # Otherwise we check whether the *path* of the parent module is inside
            # _ASYNC_IMPORT_PATH and if it is we can load it. Otherwise we leave it to the default
            # loader.
            inside_async_path = False
            for async_entry in _ASYNC_IMPORT_PATH:
                for entry in path:
                    if entry.startswith(async_entry):
                        inside_async_path = True
                        break
                if inside_async_path:
                    break
            if not inside_async_path:
                return None

        for entry in path:
            finder = importlib.machinery.FileFinder(entry, (_AsyncLoader, ('.py',)))
            spec = finder.find_spec(fullname, target=target)
            if spec:
                return spec
