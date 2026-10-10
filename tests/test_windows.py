"""Boundary and coverage tests for episode-level, future-only windows."""
import unittest

from triworld_train.windows import history_indices, window_starts


class WindowPlanningTests(unittest.TestCase):
    def test_exactly_49_frames_make_one_real_future_window(self):
        self.assertEqual(list(window_starts(49)), [1])

    def test_tail_is_anchored_to_last_48_frames(self):
        self.assertEqual(list(window_starts(66, horizon=48, stride=8)), [1, 9, 17, 18])
        last_start = list(window_starts(140, horizon=48, stride=8))[-1]
        self.assertEqual(list(range(last_start, last_start + 48)), list(range(92, 140)))

    def test_tail_anchor_is_not_duplicated(self):
        starts = list(window_starts(65, horizon=48, stride=8))
        self.assertEqual(starts, [1, 9, 17])
        self.assertEqual(len(starts), len(set(starts)))

    def test_every_real_future_frame_is_used(self):
        # Covers short episodes, non-divisible tails, and all supported strides.
        for horizon in (4, 8, 48):
            for stride in sorted({1, horizon, max(1, horizon // 2)}):
                for frame_count in range(horizon + 1, horizon + 70):
                    with self.subTest(horizon=horizon, stride=stride, count=frame_count):
                        starts = list(window_starts(frame_count, horizon=horizon, stride=stride))
                        used = {frame for start in starts for frame in range(start, start + horizon)}
                        self.assertEqual(used, set(range(1, frame_count)))
                        self.assertEqual(starts[-1] + horizon, frame_count)
                        self.assertEqual(starts, sorted(set(starts)))

    def test_history_pads_only_missing_past(self):
        self.assertEqual(list(history_indices(1, history_frames=9)), [0] * 9)
        self.assertEqual(list(history_indices(4, history_frames=9)), [0] * 6 + [1, 2, 3])
        self.assertEqual(list(history_indices(20, history_frames=9)), [0] + list(range(12, 20)))

    def test_invalid_plans_are_rejected(self):
        for kwargs in ({"horizon": 0}, {"horizon": 3}, {"stride": 0}, {"stride": 49}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                list(window_starts(100, **kwargs))


if __name__ == "__main__":
    unittest.main()
