from service_implementations.model.bigquery_base import ModelOperatorBigQueryLocation


class BigQueryDNNRegressionModelOperatorBigQueryLocation(ModelOperatorBigQueryLocation):
    model_type = "BigQuery DNN_REGRESSOR"

    def instantiate_model(self, model_id, model_version, **kwargs):
        kwargs["model_type"] = "DNN_REGRESSOR"
        super().instantiate_model(model_id, model_version, **kwargs)
