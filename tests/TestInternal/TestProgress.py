from __future__ import annotations

import io
import unittest

from msentity._internal.progress import SimpleProgress


class TestSimpleProgress(unittest.TestCase):
    def test_displays_percentage_without_third_party_package(self) -> None:
        stream = io.StringIO()
        progress = SimpleProgress(
            total=1000,
            description="Working",
            unit="items",
            min_interval=0,
            stream=stream,
        )

        progress.update(123)
        progress.close()

        output = stream.getvalue()
        self.assertIn("12.3%", output)
        self.assertIn("123/1000 items", output)
        self.assertTrue(output.endswith("\n"))

    def test_postfix_is_displayed(self) -> None:
        stream = io.StringIO()
        progress = SimpleProgress(
            total=2,
            description="Writing",
            unit="records",
            min_interval=0,
            stream=stream,
        )

        progress.update()
        progress.set_postfix({"Success": "1/1(100.0%)"})
        progress.close()

        self.assertIn("Success:1/1(100.0%)", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
