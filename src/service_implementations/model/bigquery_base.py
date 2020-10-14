from pandas import DataFrame
from google.cloud import bigquery
from mixin.bigquery_mixin import BigQueryMixin
from helpers.bigquery_location import BigQueryLocation
from service_interfaces.model_interface import ModelOperatorInterface


class ModelOperatorBigQueryLocation(ModelOperatorInterface, BigQueryMixin):
    create_model_query = """
CREATE OR REPLACE MODEL `{sql_model_path}`
OPTIONS (
{options})
AS (
    WITH train_query AS ({train_query}),
         eval_data AS ({eval_data})
    SELECT *, false as is_eval FROM train_query UNION ALL 
    SELECT *, true as is_eval FROM eval_data
)
"""

    predict_query = """
SELECT {id_column}, predicted_{target_column} FROM ML.PREDICT(MODEL `{sql_model_path}`, ({predict_query}))
ORDER BY {order}
"""

    train_metric_query = """
SELECT * FROM ML.TRAINING_INFO(MODEL `{sql_model_path}`) ORDER BY iteration
"""

    train_feature_importance_query = """
SELECT * FROM ML.FEATURE_IMPORTANCE (MODEL `{sql_model_path}`)
"""

    job_id_prefix = 'mlflow_model_'

    def __init__(self, project, dataset, _id, version, model_parameters, training_parameters):
        super().__init__(_id=_id, version=version,
                         model_parameters=model_parameters, training_parameters=training_parameters, project=project, dataset=dataset)
        self.project = project
        self.dataset = dataset
        self.client = bigquery.Client()
        self.model_id_version = f'{self.id}_{self.version}'
        self.sql_model_path = f'{self.project}.{self.dataset}.{self.model_id_version}'
        self.training_enabled = True

    def disable_training(self):
        self.training_enabled = False
        return self

    def fit(self, train_x, train_y, eval_x, eval_y) -> None:
        self.input_label_col = train_y.target_column
        model_parameters = self.get_model_parameters()
        model_parameters['data_split_method'] = 'CUSTOM'
        model_parameters['data_split_col'] = 'is_eval'
        model_parameters['input_label_cols'] = [self.input_label_col]
        quote = "'"
        options = ',\n'.join(
            [f'{key}={quote if isinstance(value, str) else ""}{value}{quote if isinstance(value, str) else ""}'
             for key, value in model_parameters.items()])
        train_data_location = BigQueryLocation(data_columns=train_x.data_columns,
                                               id_column=train_x.id_column,
                                               table=train_x.table,
                                               order=train_x.order,
                                               target_column=train_y.target_column,
                                               limit=train_x.limit)
        eval_data_location = BigQueryLocation(data_columns=eval_x.data_columns,
                                              id_column=eval_x.id_column,
                                              table=eval_x.table,
                                              order=eval_x.order,
                                              target_column=eval_y.target_column,
                                              limit=eval_x.limit)
        create_model_query = self.create_model_query.format(
            train_query=train_data_location.get_select_query(),
            eval_data=eval_data_location.get_select_query(),
            sql_model_path=self.sql_model_path,
            options=options)
        if not self.training_enabled:
            print("Training is disabled, skipping training step")
            return
        query_job = self.run_query_and_wait(
            create_model_query, job_id_prefix=self.job_id_prefix)

    def predict(self, x_uri: str, *args, **kwargs) -> None:
        target_column = self.input_label_col
        predict_query = self.predict_query.format(id_column=x_uri.id_column,
                                                  target_column=target_column,
                                                  sql_model_path=self.sql_model_path,
                                                  predict_query=x_uri.get_select_query(
                                                      include_id=True),
                                                  order=x_uri.order)
        query_job = self.run_query_and_wait(predict_query,
                                            job_id_prefix=self.job_id_prefix)
        destination = query_job.destination
        destination_location = BigQueryLocation(data_columns=[f'predicted_{target_column}'],
                                                id_column=x_uri.id_column,
                                                table=f'{destination.project}.{destination.dataset_id}.{destination.table_id}',
                                                order=x_uri.order)
        return destination_location

    def save(self, folder_path: str, *args, **kwargs) -> None:
        pass

    def load(self, folder_path: str, *args, **kwargs) -> None:
        pass

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
