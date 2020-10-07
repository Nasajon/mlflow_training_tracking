Software for training machine learning models and tracking in MLflow platform.

This software is developed to easy the work for changing the pieces around an machine learning day of work, coding the interfaces, core logging and training once and using as needed.

This project does nothing in the feature engineering/selection and there is no plan to do so. Feature engineering is a hard problem and open-world/hand-made.
Your data feature engineering/selection must be done beforehand.

MLflow Training Tracking is build around interfaces and a common __data_type__:
- __data_type__
    - Data used across the multiple interfaces. E.g.: Pandas _DataFrame_
- Data Operator Interface
    - Retrieves data from source and return as __data_type__

- Model Operator Interface
    - Receives __data_type__ retrieved from Data Interface and use to train the model then return __data_type__ after prediction on evaluation dataset

- Evaluation Metrics Operator Interface
    - Receives data y_true from Data Interface and y_pred from Model Interface, both as __data_type__. Calculate the metrics and return it

MLflow Training Tracking will by default set some tags, log every parameter and the metrics from Model Interface training and Metrics Interface.

Data Operator Interface must inherit from DataOperatorInterface and implement 
- methods
    - load_data
    - get_train_x
    - get_train_y
    - get_eval_x
    - get_eval_y

Model Operator Interface must inherit from ModelOperatorInterface and implement 

- properties
    - model_type
- methods
    - instantiate_model
    - fit
    - predict
    - save
    - load
    - get_train_metrics

Evaluation Metrics Operator Interface must inherit from EvaluationMetricsOperatorInterface and implement
- methods
    - load_data
    - get_eval_metrics

Regression Model
- methods
    - explained_variance_score
    - mean_absolute_error
    - mean_squared_error
    - median_absolute_error
    - r2_score
    - max_error

Classification Model
- methods
    - TBD


Current state of implemented interfaces:
```
service_implementations
```
- Data Operator Interface
    ```
    data
    ```
    - CSV to DataFrame
        ```
        csv_to_dataframe.FileToDataFrame
        ```
        - Receives path to file and load it into a DataFrame
    - BigQuery to DataFrame
        ```
        bigquery_to_dataframe.BigQueryToDataFrame
        ```    
        - Receives SQL Queries, run query and load it into a DataFrame
    - BigQuery Location (to BigQuery Location)
        ```
        bigquery_location.DataOperatorBigQueryLocation
        ```     
        - Receives a dictionary that identify a table
        - Dictionary structure:
        ```
        {
            'columns': ['feature_column1', 'feature_column2', 'target_column'], #BigQuery columns SELECT clause
            'id_column': 'id_column', #used to identify an row
            'table': 'project.database.table', #BigQuery table FROM clause
            'order': 'id_column', #BigQuery ORDER BY clause, BQ does not have natural order, we need to enforce order to get the same results over multiple queries
            'limit': 7300 #BigQuery LIMIT clause
        }
        ```

- Model Operator Interface
BigQueryDNNRegressionModelOperatorBigQueryLocation
    ```
    model
    ```
    - BigQuery DNN Regression (to BigQuery Location)
        ```
        bigquery_dnn_regression.BigQueryDNNRegressionModelOperatorBigQueryLocation
        ```
        - receives BigQuery Location and train a DNN model using BigQuery. See [BigQuery manual](https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create-dnn-models) for parameters
    - BigQuery XGB Tree Regression (to BigQuery Location)
        ```
        bigquery_xgb_regression.BigQueryXGBRegressionModelOperatorBigQueryLocation
        ```    
        - Receives BigQuery Location and train a XGB model using BigQuery. See [BigQuery manual](https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create-boosted-tree) for parameters
    - Keras DNN Regression (to DataFrame)
        ```
        keras_dnn_regression.KerasRegressionModelOperatorDataFrame
        ```      
        - Receives DataFrame and train a DNN model using Keras. See [Keras manual](https://keras.io/api/layers/) for parameters
    - XGB Tree Regression (to DataFrame)
        ```
        xgb_regression.XGBRegressionModelOperatorDataFrame
        ```      
        - Receives DataFrame and train a XGB Tree model using XGB. See [XGB manual](https://xgboost.readthedocs.io/en/latest/parameter.html) for parameters.

- Evaluation Metrics Operator Interface
    ```
    evaluation_metrics
    ```
    - Regression
        - BigQuery Location (to NumPy Array)
            ```
            bigquery_location.EvaluationRegressionMetricsBigQueryLocationNumpyArray
            ```         
            - Receives BigQuery Location, query it and get result as Numpy Array, then calculate the metrics locally
        - NumpyArray (to NumPy Array)
            ```
            numpy_array.EvaluationRegressionMetricsNumpyArray
            ```         
            - Receives Numpy/DataFrame, and calculate the metrics
    - Classification
        - None

Known issues:
- BigQuery Location is not SQL-injection-free, and I don't care

Future work:
- Add test dataset

