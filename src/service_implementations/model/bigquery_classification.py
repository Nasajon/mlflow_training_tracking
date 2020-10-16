from pandas import DataFrame
from google.cloud import bigquery
from mixin.bigquery_mixin import BigQueryMixin
from helpers.bigquery_location import BigQueryLocation
from service_implementations.model.bigquery_base import ModelOperatorBigQueryLocation


class ClassificationModelOperatorBigQueryLocation(ModelOperatorBigQueryLocation):

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

        return metrics_struct_list
