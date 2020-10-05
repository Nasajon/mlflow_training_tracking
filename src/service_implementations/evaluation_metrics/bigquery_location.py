from service_interfaces.evaluation_metrics_interface import EvaluationMetricsOperatorInterface
from service_implementations.evaluation_metrics.numpy_array import EvaluationRegressionMetricsNumpyArray
from mixin.bigquery_mixin import BigQueryMixin
import numpy as np
from google.cloud import bigquery


class EvaluationRegressionMetricsBigQueryLocationNumpyArray(EvaluationRegressionMetricsNumpyArray, BigQueryMixin):
    def __init__(self):
        self.client = bigquery.Client()
        super().__init__()

    def load_data(self, y_true, y_pred, *args, **kwargs):
        self.y_true = self.load_dataframe_from_query(query=y_true.get_select_query(),
                                                     job_id_prefix='mlflow_metrics_').to_numpy()
        self.y_pred = self.load_dataframe_from_query(query=y_pred.get_select_query(),
                                                     job_id_prefix='mlflow_metrics_').to_numpy()
