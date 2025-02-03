### PupAlz

## Description
Code to process and analyze data from the PupAlz project. Pupil dilation 
data is collected using a Tobii eyetracker and E-prime. Pupil data was
collected during several tasks, including digit span, verbal fluency, HVLT, 
stroop, and auditory oddball.

## Code
Scripts are organized by task. Each task has a script to process subject data
and a script to analyze group data. Subject scripts should be run first followed
by group scripts. The `pupil_utils.py` script contains functions used by all scripts.

### Scripts to process tasks

- **[digitspan_proc_subject.py](digitspan_proc_subject.py)**: Processes individual subject data for the Digit Span task.
- **[digitspan_proc_group.py](digitspan_proc_group.py)**: Analyzes group data for the Digit Span task.
- **[digitspan_concat_subjects.py](digitspan_concat_subjects.py)**: Concatenates individual subject data for the Digit Span task.
- **[fluency_proc_subject.py](fluency_proc_subject.py)**: Processes individual subject data for the Fluency task.
- **[fluency_proc_group.py](fluency_proc_group.py)**: Analyzes group data for the Fluency task.
- **[hvlt_encoding_proc_subject.py](hvlt_encoding_proc_subject.py)**: Processes individual subject data for the encoding phase of HVLT task.
- **[hvlt_encoding_proc_group.py](hvlt_encoding_proc_group.py)**: Analyzes group data for the encoding phase of HVLT task.
- **[hvlt_recognition_proc_subject.py](hvlt_recognition_proc_subject.py)**: Processes individual subject data for the recognition phase of HVLT task.
- **[hvlt_recognition_proc_group.py](hvlt_recognition_proc_group.py)**: Analyzes group data for the recognition phase of HVLT task.
- **[hvlt_recall_proc_subject.py](hvlt_recall_proc_subject.py)**: Processes individual subject data for the recall phase of HVLT task.
- **[hvlt_recall_proc_group.py](hvlt_recall_proc_group.py)**: Analyzes group data for the recall phase of HVLT task.
- **[oddball_setup_subject.py](oddball_setup_subject.py)**: Sets up subject directory and recodes data for the Oddball task.
- **[oddball_proc_subject.py](oddball_proc_subject.py)**: Processes individual subject data for the Oddball task.
- **[oddball_proc_group.py](oddball_proc_group.py)**: Analyzes group data for the Oddball task.
- **[stroop_proc_subject.py](stroop_proc_subject.py)**: Processes individual subject data for the Stroop task.
- **[stroop_proc_group.py](stroop_proc_group.py)**: Analyzes group data for the Stroop task.

### Additional scripts
- **[calculate_ach_burden.py](calculate_ach_burden.py)**: Calculate anti-cholinergic burden score based on list of medications.
- **[pupil_utils.py](pupil_utils.py)**: Contains utility functions used by all task scripts to process data.