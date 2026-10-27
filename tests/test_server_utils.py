"""Unit tests for server startup and readiness polling."""

import shlex
import sys
import time
import unittest

from llm_optimizer.exceptions import ServerNotReadyError
from llm_optimizer.server_utils import start_server

# Nothing listens on the discard port, so readiness polling never succeeds.
UNREACHABLE_URL = "http://127.0.0.1:9/health"


def _python_cmd(code: str) -> str:
    return shlex.join([sys.executable, "-c", code])


class TestStartServer(unittest.TestCase):
    def test_fails_fast_when_server_process_exits(self):
        start = time.monotonic()
        with self.assertRaises(ServerNotReadyError) as ctx:
            start_server(
                _python_cmd("import sys; sys.exit(3)"),
                {},
                UNREACHABLE_URL,
                mute=True,
                timeout=60,
            )
        self.assertLess(time.monotonic() - start, 15)
        self.assertIn("exited with code 3", str(ctx.exception))

    def test_times_out_when_server_never_becomes_ready(self):
        start = time.monotonic()
        with self.assertRaises(ServerNotReadyError) as ctx:
            start_server(
                _python_cmd("import time; time.sleep(60)"),
                {},
                UNREACHABLE_URL,
                mute=True,
                timeout=1,
            )
        self.assertLess(time.monotonic() - start, 15)
        self.assertIn("did not become ready", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
