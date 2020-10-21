from pandas import DataFrame
from google.cloud import bigquery
from mlflow_training_tracking.mixin.bigquery_mixin import BigQueryMixin
from mlflow_training_tracking.helpers.bigquery_location import BigQueryLocation
from mlflow_training_tracking.service_implementations.model.bigquery_base import ModelOperatorBigQueryLocation


class RegressionModelOperatorBigQueryLocation(ModelOperatorBigQueryLocation):
    prediction_column_name = "predicted_{target_column}"
    predict_query = """
SELECT {{id_column}}, {prediction_column_name} FROM ML.PREDICT(MODEL `{{sql_model_path}}`, ({{predict_query}}))
ORDER BY {{order}}
""".format(prediction_column_name=prediction_column_name)
