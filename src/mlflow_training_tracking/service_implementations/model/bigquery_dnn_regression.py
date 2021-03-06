from mlflow_training_tracking.service_implementations.model.bigquery_regression import RegressionModelOperatorBigQueryLocation


class BigQueryDNNRegressionModelOperatorBigQueryLocation(RegressionModelOperatorBigQueryLocation):
    model_type = "BigQuery DNN_REGRESSOR"

    def __init__(self, model_parameters, **kwargs):
        model_parameters["model_type"] = "DNN_REGRESSOR"
        super().__init__(model_parameters=model_parameters, **kwargs)
