from mlflow_training_tracking.service_implementations.model.bigquery_classification import ClassificationModelOperatorBigQueryLocation


class BigQueryDNNClassificationModelOperatorBigQueryLocation(ClassificationModelOperatorBigQueryLocation):
    model_type = "BigQuery DNN_CLASSIFIER"

    def __init__(self, model_parameters, **kwargs):
        model_parameters["model_type"] = "DNN_CLASSIFIER"
        super().__init__(model_parameters=model_parameters, **kwargs)
