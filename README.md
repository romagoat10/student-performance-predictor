# Student Performance Predictor

A beginner-friendly machine-learning project built with Python and Streamlit. Upload a CSV, select a numeric outcome and input columns, then fit a linear regression model and compare its test-set performance with a mean-only baseline.

The built-in practice dataset is generated synthetically. It is for learning the interface and workflow only; it is not real student data and should not be used to draw conclusions about students.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder and install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   python -m streamlit run app.py
   ```

## CSV format

Provide at least 10 rows, a numeric outcome column, and one or more input columns. Inputs may be numeric or categorical. For example:

```csv
study_hours,attendance_percent,previous_score,final_score
6,92,78,81
3,75,65,69
10,98,88,94
```

Select the outcome and input columns in the sidebar. Do not select ID columns or variables that would only be known after the outcome happens. Missing numeric inputs are filled with the median and missing categories with the most common value. The app makes a random train/test split and reports MAE, RMSE, R², and the error from a simple mean-only baseline.

## Important limitations

- A single train/test split is only a first evaluation; scores can vary with the data and split.
- Prediction errors are in the same units as the outcome.
- Relationships in a dataset do not show that an input causes an outcome.
- Do not use this educational demo to grade, admit, rank, or otherwise make decisions about real students.
