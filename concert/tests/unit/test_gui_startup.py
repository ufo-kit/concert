"""GUI opt-in and helpers must also be testable without any Qt dependency."""
import argparse
import asyncio
import subprocess
import sys
from unittest.mock import Mock

import pytest

from concert.gui import with_signals
from concert.session.utils import StartCommand


def test_gui_is_opt_in():
    parser = argparse.ArgumentParser()
    option = StartCommand().opts['--with-gui']
    parser.add_argument('--with-gui', **option)
    assert parser.parse_args([]).with_gui is False
    assert parser.parse_args(['--with-gui']).with_gui is True


def test_import_does_not_initialize_qt():
    result = subprocess.run([sys.executable, '-c', '''
import sys
import concert.gui
from concert.session.utils import StartCommand
assert not any(name.startswith(('PyQt6', 'pyqtgraph', 'qasync')) for name in sys.modules)
'''], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_missing_dependencies_message():
    result = subprocess.run([sys.executable, '-c', '''
import sys
sys.modules['PyQt6'] = None
from concert.gui import get_application
try:
    get_application()
except RuntimeError as error:
    assert 'concert[gui]' in str(error)
else:
    raise AssertionError('Expected a helpful dependency error')
'''], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_signal_decorator_awaits_and_returns_result():
    started, finished, error = Mock(), Mock(), Mock()

    @with_signals(started, finished, error)
    async def operation(value):
        return value

    started.emit.assert_not_called()
    assert asyncio.run(operation(42)) == 42
    started.emit.assert_called_once_with()
    finished.emit.assert_called_once_with()
    error.emit.assert_not_called()


def test_signal_decorator_reports_and_propagates_error():
    finished, error = Mock(), Mock()

    @with_signals(finish_signal=finished, exception_signal=error)
    async def operation():
        raise ValueError('failed')

    with pytest.raises(ValueError, match='failed'):
        asyncio.run(operation())
    error.emit.assert_called_once()
    finished.emit.assert_called_once_with()
