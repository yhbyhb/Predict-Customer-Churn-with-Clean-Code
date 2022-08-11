# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project identifies credit card customers that are most likely to chrun. 

## Files and data description
* `churn_library.py` - a library of functions to find customers who are likely to churn
* `churn_script_logging_and_test.py` - contains unit tests of `churn_library.py`
* `data/bank_data.csv` - a data file for this project
* `images/` - results images of EDA and plot results of trained model.
* `logs/churn_library.log` - log file for unit tests.
* `models/` - trained models for predcition.

## Running Files
Before running files, install dependencies as following.
```
$ python -m pip install -r requirements_py3.8.txt
```

To get prediction results, execute `churn_library`.
```
$ python churn_library.py
```

For unit tests, run `churn_script_logging_and_tests.py` or pytest it.
```
$ python churn_script_logging_and_tests.py
```