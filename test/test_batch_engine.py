import os
import sys
import unittest
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from batch_engine import (  # noqa: E402
    process_pool_thread_workers,
    run_indexed_threaded_batches_in_process_pool_bounded,
)


def exit_process_for_test(item):
    if item == "exit":
        os._exit(3)
    return item


class ProcessPoolBatchEngineTests(unittest.TestCase):
    def test_process_pool_thread_workers_avoids_nested_parallel_fanout(self):
        self.assertEqual(
            1,
            process_pool_thread_workers(
                process_workers=8,
                requested_thread_workers=16,
            ),
        )
        self.assertEqual(
            16,
            process_pool_thread_workers(
                process_workers=1,
                requested_thread_workers=16,
            ),
        )

    def test_bounded_pool_reports_broken_submit_as_batch_failure(self):
        results = list(
            run_indexed_threaded_batches_in_process_pool_bounded(
                [(1, ["exit"]), (2, ["after-exit"])],
                group_count=2,
                run_item=exit_process_for_test,
                process_workers=1,
                thread_workers=1,
                max_pending_batches=1,
            )
        )

        self.assertEqual([1, 2], [result.index for result in results])
        self.assertIsInstance(results[0].error, BrokenProcessPool)
        self.assertIsInstance(results[1].error, BrokenProcessPool)


if __name__ == "__main__":
    unittest.main()
