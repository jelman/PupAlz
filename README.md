### PupAlz

## Description
Code to process and analyze data from the PupAlz project. Pupil dilation 
data is collected using a Tobii eyetracker and E-prime. Pupil data was
collected during several tasks, including digit span, verbal fluency, HVLT, 
stroop, and auditory oddball.

## Prerequisites
1. Install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Clone this repository
```bash
git clone https://github.com/jelman/PupAlz.git
cd PupAlz
```
3. Create and activate the conda environment
```bash
conda env create -f environment.yml
conda activate pupalz
```

**Note:** Always activate your environment before running any scripts. The code is compatible with both Python 2 and Python 3, but Python 3.6+ is recommended.

## Data organization

### Input data
Input data is expected to be under a folder named `Raw Pupil Data/` and organized under `Timepoint N/` folders. These timepoints specific folders are used to verify the session in each data file. 

**Supported file formats:** `.gazedata`, `.csv`, `.xlsx`

```
└── [project folder]/
    ├── Raw Pupil Data/
    │   └── [task name] (optional)/
    │       ├── Timepoint 1/
    │       │   └── [taskname]-[subjectid]-1.(gazedata/xlsx)
    │       ├── Timepoint 2/
    │       │   └── [taskname]-[subjectid]-2.(gazedata/xlsx)
    │       └── Timepoint N/
    │           └── [taskname]-[subjectid]-[N].(gazedata/xlsx)
    └── Processed Pupil Data/
        └── [task name] (optional)/
```

**Subject ID Format:**
Subject IDs should follow this format:
- Any number of digits (at least one)
- Followed by a dash (`-`)
- Followed by a single digit

Examples: `12345-1`, `987-2`, `1001-3`

(Regular expression: `\d+-\d`)

**Data Processing Notes:**
- Scripts automatically handle blink artifact removal and filtering
- Pupil data is resampled to 1-second intervals for most tasks  
- Samples with >50% blinks are automatically excluded
- Output includes both individual trial data and averaged condition data

### Output data

Processed data files are automatically saved to the `Processed Pupil Data/` directory structure. The output filename is created by:
1. Replacing `Raw Pupil Data` with `Processed Pupil Data` in the file path
2. Stripping off the original filename extension (e.g., `.gazedata`, `.xlsx`)
3. Appending `_ProcessedPupil.csv`

**Example:**
- Input: `Raw Pupil Data/Timepoint 1/digitspan-12345-1.gazedata`
- Output: `Processed Pupil Data/Timepoint 1/digitspan-12345-1_ProcessedPupil.csv`

The processed files contain cleaned, filtered, and summarized pupil dilation data at 1-second intervals for each trial condition.

**Additional Output Files:**
Some tasks generate additional output files:
- `*_PupilPlot.png`: Visualization of processed pupil data
- `*_SessionData.csv`: Session-level summary data (for oddball/stroop tasks)
- `*_PSTCdata.csv`: Peristimulus time course data (for oddball/stroop tasks)
- `*_Quartiles.csv` or `*_Tertiles.csv`: Time-blocked averages for some tasks

## Overview of scripts
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

## Processing steps

**Important:** Always activate your conda environment before running scripts:
```bash
conda activate pupalz
```

### Basic Processing Workflow
1. **Process individual subject data** for each task. Scripts will open a file selection window to select all raw subject data files. Processed data is automatically saved to the `Processed Pupil Data/` directory.
```bash
python <task_name>_proc_subject.py
```

2. **Process group data**. Scripts will open a folder selection window to select the directory containing all processed subject data. Group analyses and summaries are saved to the same directory.
```bash
python <task_name>_proc_group.py
```

### Command Line Usage
You can also run scripts from command line with file arguments:
```bash
# Process specific files
python digitspan_proc_subject.py path/to/file1.gazedata path/to/file2.gazedata

# Process group data from directory
python digitspan_proc_group.py path/to/processed/data/directory
```

### Special Cases
- **Oddball task**: Run `oddball_setup_subject.py` first to recode data before processing
- **Stroop task**: Requires corresponding E-Prime files in the same directory
- **HVLT tasks**: Some group scripts require QC files for excluding problematic data

## Troubleshooting

**Common Issues:**
- **"Could not find valid subject ID"**: Check that filenames follow the `[digits]-[digit]` format
- **Import errors**: Ensure conda environment is activated
- **File not found**: Verify file paths and directory structure match expected format
- **E-Prime file errors (Stroop)**: Ensure E-Prime files are converted to CSV and in same directory

**Getting Help:**
- Each script shows usage information when run without arguments
- Check console output for specific error messages
- Verify input data format matches expected structure

## Quality Control
- Scripts include built-in quality control measures
- Visual plots are generated for each processed file
- Group processing scripts can accept QC files to exclude problematic data
- QC files should contain columns: `Subject`, `Session`/`Timepoint`, `Exclude` (0/1)

