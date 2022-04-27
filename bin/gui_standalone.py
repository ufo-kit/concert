from concert.devices.motors.dummy import LinearMotor
from concert.gui.parameterizable import ParameterizableWidget
from qasync import QApplication
import qasync
import asyncio
import sys


async def main():
    def close_future(future, loop):
        loop.call_later(10, future.cancel)
        future.cancel()

    loop = asyncio.get_event_loop()
    future = asyncio.Future()

    app = QApplication.instance()

    lin = LinearMotor()
    lin_widget = ParameterizableWidget(lin)
    lin_widget.show()

    await future
    return True


if __name__ == "__main__":
    try:
        qasync.run(main())
    except asyncio.exceptions.CancelledError:
        sys.exit(0)
