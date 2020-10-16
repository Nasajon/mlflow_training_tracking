from mlflow_training_tracking.service_interfaces.evaluation_metrics_interface import EvaluationMetricsOperatorInterface
from mlflow_training_tracking.service_implementations.evaluation_metrics.numpy_array import *
from mlflow_training_tracking.mixin.bigquery_mixin import BigQueryMixin
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


class EvaluationClassificationMetricsBigQueryLocationNumpyArray(EvaluationBinaryClassificationMetricsNumpyArray, BigQueryMixin):
    def __init__(self, probability_threshold):
        self.client = bigquery.Client()
        super().__init__(probability_threshold=probability_threshold)

    def load_data(self, y_true, y_pred, *args, **kwargs):
        y_true = self.load_dataframe_from_query(query=y_true.get_select_query(include_id=True),
                                                job_id_prefix='mlflow_metrics_')
        y_pred = self.load_dataframe_from_query(query=y_pred.get_select_query(include_id=True),
                                                job_id_prefix='mlflow_metrics_')
        super().load_data(y_true=y_true, y_pred=y_pred)
