import unittest
from contextlib import ExitStack
from unittest.mock import patch

import pandas as pd

import oddball_proc_subject
import stroop_proc_subject


class SubjectConditionConcatTest(unittest.TestCase):
    def test_condition_frames_are_combined_in_order(self):
        cases = [
            (oddball_proc_subject, ['Target', 'Standard'], ['Standard', 'Target']),
            (stroop_proc_subject, ['Congruent', 'Incongruent', 'Neutral'],
             ['Congruent', 'Incongruent', 'Neutral']),
        ]
        for module, reshape_order, expected_order in cases:
            with self.subTest(module=module.__name__), ExitStack() as stack:
                raw = pd.DataFrame({
                    'Subject': ['123-1'], 'Session': [1], 'CurrentObject': ['Stimulus']
                })
                resampled = pd.DataFrame({
                    'CRESP': [5], 'DiameterPupilLRFilt': [1.0], 'BlinksLR': [0],
                    'TrialId_x': [1], 'TrialId_y': [1],
                })
                session = pd.DataFrame({
                    'Condition': ['Target', 'Standard', 'C', 'I', 'N'],
                    'Timestamp': range(5),
                })
                eprime = pd.DataFrame(columns=['ExperimentName', 'Subject', 'Session'])
                frames = [
                    pd.DataFrame({'Condition': [condition], 'Timepoint': [0.5],
                                  'Dilation': [i]}, index=[10 + i])
                    for i, condition in enumerate(reshape_order)
                ]

                def mock(obj, name, **kwargs):
                    return stack.enter_context(patch.object(obj, name, **kwargs))

                reads = [raw, eprime] if module is stroop_proc_subject else [raw]
                mock(module.pd, 'read_csv', side_effect=reads)
                mock(module.pupil_utils, 'get_subid', return_value='123-1')
                mock(module.pupil_utils, 'get_timepoint', return_value='1')
                mock(module.pupil_utils, 'deblink', side_effect=lambda df: df)
                mock(module.pupil_utils, 'resamp_filt_data', return_value=resampled)
                mock(module.pupil_utils, 'plot_qc')
                mock(module.pupil_utils, 'zscore', side_effect=lambda values: values)
                mock(module.pupil_utils, 'get_outfile', return_value='unused.csv')
                mock(module.pupil_utils, 'get_proc_outfile', return_value='unused.csv')
                mock(module, 'get_sessdf', return_value=session)
                mock(module, 'get_blink_pct', return_value=0)
                mock(module, 'proc_all_trials', return_value=[None] * len(frames))
                mock(module, 'reshape_df', side_effect=frames)
                mock(module, 'ts_glm', return_value={})
                mock(module, 'save_glm_results')
                plot = mock(module, 'plot_pstc')
                save = mock(module, 'save_pstc')
                mock(pd.DataFrame, 'to_csv')
                if module is stroop_proc_subject:
                    mock(module, 'get_eprime_fname', return_value='eprime.csv')

                module.proc_subject(['input.csv'])

                combined = plot.call_args.args[0]
                self.assertEqual(combined['Condition'].tolist(), expected_order)
                self.assertEqual(combined['Dilation'].tolist(),
                                 [reshape_order.index(condition) for condition in expected_order])
                self.assertEqual(combined.index.tolist(), list(range(len(frames))))
                self.assertEqual(combined['Subject'].tolist(), ['123-1'] * len(frames))
                self.assertEqual(combined['Session'].tolist(), ['1'] * len(frames))
                pd.testing.assert_frame_equal(combined, save.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
