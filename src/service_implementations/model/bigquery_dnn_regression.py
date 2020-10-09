from service_implementations.model.bigquery_base import ModelOperatorBigQueryLocation


class BigQueryDNNRegressionModelOperatorBigQueryLocation(ModelOperatorBigQueryLocation):
    model_type = "BigQuery DNN_REGRESSOR"

    def __init__(self, model_parameters, **kwargs):
        model_parameters["model_type"] = "DNN_REGRESSOR"
        super().__init__(model_parameters=model_parameters, **kwargs)
