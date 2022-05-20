"""Internal module for the support of imports with top-level `await' (outside of `async def'
functions).
"""
import ast
import asyncio
import importlib
import inspect
import logging
import os
import sys
import threading
import concert.session.management as cs
from concert.config import AIODEBUG


LOG = logging.getLogger(__name__)
SESSION_PATH = cs.path()


class AsyncLoader(importlib.machinery.SourceFileLoader):
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
        # *nodes* are all ast nodes of the module
        nodes = ast.parse(self.get_data(self.path)).body
        eval_nodes(nodes, module.__dict__, filename=self.path)


class AsyncMetaPathFinder(importlib.abc.MetaPathFinder):
    """
    MetaPathFinder implementation which used :class;`.FileFinder` for dealing with paths and
    instantiated with our AsyncLoader for the handling of `await' outside of `async def' functions.
    """
    def find_spec(self, fullname, path, target=None):

        if path is None:
            # If this is a top-level import *path* is None, otherwise do not mess with it and use
            # the one provided by the package hierarchy.
            # If it is None, make sure user modules are found before system modules, e.g. when
            # someone decides to name a session `test', which exists in pythonX.Y/test/__init__.py
            path = [SESSION_PATH, os.getcwd()] + sys.path

        for entry in path:
            finder = importlib.machinery.FileFinder(entry, (AsyncLoader, ('.py',)))
            spec = finder.find_spec(fullname, target=target)
            if spec:
                return spec


def eval_nodes(nodes, module_dict, filename=''):
    """
    Source code in ast *nodes* is split into import and non-import statements. All import statements
    are executed one-by-one and all other statements are aggregated and executed together. This way
    we allow nesting of module imports which have top-level `await' keywords. This is possible
    thanks to the fact that when we call :func:`eval` on an import statement, it will not return a
    coroutine but it will invoke the import mechanism (thus us) on the module to be imported.  Then
    that nested module is processed with the loop run if necessary and stopped. Then the recursion
    ends and the importing module processes the rest of the statements with possible loop runs which
    is fine becuase the nested module is processed already, so no nested loop runs are attempted.
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
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
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
                LOG.log(
                    AIODEBUG,
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
