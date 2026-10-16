# Student Performance Analysis and Prediction

## Project overview
This project explores the UCI Student Performance dataset and trains a multilayer perceptron (MLP) to classify whether a student passes the final-grade threshold.

## Target definition
`pass = 1` when final grade `G3 >= 10`, otherwise `pass = 0`. This is a project-defined threshold, not a guarantee about every institution's grading rules.

## Dataset
The script downloads the UCI Student Performance archive and uses `student-mat.csv` (Mathematics subject). The dataset has 395 records and 33 original columns in the commonly distributed file; verify the actual shape printed by your run.

Dataset page: https://archive.ics.uci.edu/dataset/320/student+performance

## Run
Recommended: upload `student_performance_eda.py` to Google Colab, or run in an environment with Python and internet access.

Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

Then run:
```bash
python student_performance_eda.py
```

## Outputs
The script creates `outputs/` with EDA figures, model evaluation plots, statistics, feature importance, and `test_metrics.csv`.

## Important notes
- Do not claim results before running the code.
- Model results depend on the run and the random seed.
- This is a small, observational dataset. Predictions are not causal explanations and should not be used to make high-stakes decisions about students.
- `G1` and `G2` are included as prior grades and can be predictive of `G3`; explain this when interpreting feature importance.
