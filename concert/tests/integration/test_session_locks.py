import tempfile
import os
from concert import get_canonical_version
from concert.coroutines.base import background
from concert.session.utils import StartCommand, SessionLockException, DeviceLockException
from concert.session.management import _SESSION_TEMPLATE
from concert.tests import TestCase


class TestSessionLocks(TestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.session1 = tempfile.NamedTemporaryFile('w+t', delete=False, suffix=".py")
        self.session2 = tempfile.NamedTemporaryFile('w+t', delete=False, suffix=".py")
        with open(self.session1.name, 'w') as session_file:
            session_file.write(_SESSION_TEMPLATE.format(version=get_canonical_version(), doc="test_session1"))

        with open(self.session2.name, 'w') as session_file:
            start_sleep_time = 0  # s
            long_running_session = _SESSION_TEMPLATE + f"\nimport time\ntime.sleep({start_sleep_time})"
            session_file.write(long_running_session.format(version=get_canonical_version(), doc="test_session2"))

    def tearDown(self):
        os.remove(self.session1.name)
        os.remove(self.session2.name)

    async def test_single_session_start(self):
        """
        Test if starting a single session works
        """
        start_command = StartCommand()
        start_command.run(filename=self.session1.name, loglevel="info", non_interactive=True)

    async def test_session_lock(self):
        """
        Test if starting the same session fails
        """
        start_command1 = StartCommand()
        start_command2 = StartCommand()

        @background
        async def run_session(command):
            command.run(filename=self.session2.name, loglevel="info", non_interactive=True)

        f = run_session(start_command1)
        with self.assertRaises(SessionLockException):
            await  run_session(start_command2)
        await f


class TestDeviceLocks(TestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.session1 = tempfile.NamedTemporaryFile('w+t', delete=False, suffix=".py")
        self.session2 = tempfile.NamedTemporaryFile('w+t', delete=False, suffix=".py")
        self.session3 = tempfile.NamedTemporaryFile('w+t', delete=False, suffix=".py")
        self.session4 = tempfile.NamedTemporaryFile('w+t', delete=False, suffix=".py")
        with open(self.session1.name, 'w') as session_file:
            session_with_device_a = _SESSION_TEMPLATE + f"\nDEVICE_LOCKS=['a']"
            session_file.write(session_with_device_a.format(version=get_canonical_version(), doc="test_session_a"))

        with open(self.session2.name, 'w') as session_file:
            session_with_device_b = _SESSION_TEMPLATE + f"\nDEVICE_LOCKS=['b']"
            session_file.write(session_with_device_b.format(version=get_canonical_version(), doc="test_session_b"))

        with open(self.session3.name, 'w') as session_file:
            session_with_device_x_y = _SESSION_TEMPLATE + f"\nDEVICE_LOCKS=['x', 'y']"
            session_file.write(session_with_device_x_y.format(version=get_canonical_version(), doc="test_session_x_y"))

        with open(self.session4.name, 'w') as session_file:
            session_with_device_y_z = _SESSION_TEMPLATE + f"\nDEVICE_LOCKS=['y', 'z']"
            session_file.write(session_with_device_y_z.format(version=get_canonical_version(), doc="test_session_y_z"))

    def tearDown(self):
        os.remove(self.session1.name)
        os.remove(self.session2.name)
        os.remove(self.session3.name)
        os.remove(self.session4.name)

    async def test_independent_locks(self):
        """
        Tests if two session with device locks that are not overlapping can be started.
        """
        start_command1 = StartCommand()
        start_command2 = StartCommand()

        @background
        async def run_session(command, session_file):
            command.run(filename=session_file, loglevel="info", non_interactive=True)

        f = run_session(start_command1, self.session1.name)

        await  run_session(start_command2, self.session2.name)
        await f

    async def test_locks(self):
        """
        Tests if it fails starting two session where device locks are not overlapping.
        """
        start_command1 = StartCommand()
        start_command2 = StartCommand()

        @background
        async def run_session(command, session_file):
            command.run(filename=session_file, loglevel="info", non_interactive=True)

        f = run_session(start_command1, self.session3.name)

        with self.assertRaises(DeviceLockException):
            await  run_session(start_command2, self.session4.name)
            await f
