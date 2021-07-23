from concert.commands import create_command, command, COMMANDS
from concert.tests import TestCase


async def corofunc():
    return 1


class TestCommands(TestCase):
    def test_create_command(self):
        cmd = create_command(corofunc)
        self.assertEqual(cmd(), 1)

    def test_command(self):
        command(name='foo')(corofunc)
        self.assertTrue('foo' in COMMANDS)
        self.assertEqual(COMMANDS['foo'](), 1)
