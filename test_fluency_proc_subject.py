import unittest

import pandas as pd

from fluency_proc_subject import get_trial_events


class TrialEventsTest(unittest.TestCase):
    def test_trial_events_with_supported_trial_counts(self):
        for count in (4, 6):
            with self.subTest(count=count):
                objects = ['ReadLetter', 'BeginFile', 'RecordLetter'] * count
                df = pd.DataFrame({
                    'TETTime': range(len(objects)),
                    'CurrentObject': objects,
                })

                events = get_trial_events(df)

                self.assertEqual(len(events), count * 4)
                self.assertEqual(events['Trial'].tolist(),
                                 [trial for trial in range(1, count + 1) for _ in range(4)])
                self.assertEqual(events['Condition'].tolist(),
                                 ['Letter'] * (count * 2) + ['Category'] * (count * 2))
                self.assertTrue(events.index.is_monotonic_increasing)


if __name__ == '__main__':
    unittest.main()
