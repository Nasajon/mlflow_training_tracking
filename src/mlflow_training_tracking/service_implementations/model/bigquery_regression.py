from pandas import DataFrame
from google.cloud import bigquery
from mlflow_training_tracking.mixin.bigquery_mixin import BigQueryMixin
from mlflow_training_tracking.helpers.bigquery_location import BigQueryLocation
from mlflow_training_tracking.service_implementations.model.bigquery_base import ModelOperatorBigQueryLocation


class RegressionModelOperatorBigQueryLocation(ModelOperatorBigQueryLocation):

    def get_train_metrics(self, *args, **kwargs) -> 'list[dict]':
        """Return a list of dictionaries to log in metrics MLFlow server

        Returns:
            list[dict]: E.g.
            [
                {
                metrics: {
                    validadtion_1.metric_1: value,
                    validadtion_1.metric_2: value,
                    validadtion_2.metric_1: value,
                    validadtion_2.metric_2: value
                    },
                step: 1
                },
                                {
                metrics: {
                    validadtion_1.metric_1: value,
                    validadtion_1.metric_2: value,
                    validadtion_2.metric_1: value,
                    validadtion_2.metric_2: value
                    },
                step: 2
                }
            ]
        """
        metrics_struct_list = []
        train_metric_query = self.train_metric_query.format(
            sql_model_path=self.sql_model_path)
        query_job = self.run_query_and_wait(train_metric_query,
                                            job_id_prefix=self.job_id_prefix)
        rows = query_job.result()

        for row in rows:
            row_metric = {
                'loss': row.loss,
                'eval_loss': row.eval_loss,
                'duration_ms': row.duration_ms,
                'learning_rate': row.learning_rate
            }
            metrics_struct_list.append({
                'metrics': row_metric,
                'step': row.iteration
            })

        return metrics_struct_list
