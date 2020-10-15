from copy import deepcopy
from google.cloud import bigquery
from service_interfaces.data_interface import DataOperatorInterface
from helpers.bigquery_location import BigQueryLocation
from helpers.default_settings import set_defaults_save_job_config


class DataOperatorBigQueryLocation(DataOperatorInterface):

    def __init__(self, train_uri: str, eval_uri: str, target_column: str,
                 row_id_column: str, predictions_dataset: str,
                 save_data_job_config=None):
        super().__init__(
            train_uri, eval_uri, target_column, row_id_column,
            predictions_dataset=predictions_dataset,
            save_data_job_config='custom' if save_data_job_config is not None else None
        )
        self.predictions_dataset = predictions_dataset
        self.save_data_job_config = save_data_job_config
        self.client = bigquery.Client()
        self.job_pool = list()

    def load_data(self) -> DataOperatorInterface:
        train_data = deepcopy(self.train_uri)
        eval_data = deepcopy(self.eval_uri)

        self.train_y = train_data.copy()
        self.train_y.update(
            {'data_columns': []})
        self.train_x = train_data.copy()

        # eval_data['columns'].remove(self.target_column)
        self.eval_y = eval_data.copy()
        self.eval_y.update(
            {'data_columns': []})
        self.eval_x = eval_data.copy()

        return self

    def get_train_x(self) -> str:
        return BigQueryLocation(**self.train_x, id_column=self.row_id_column)

    def get_train_y(self) -> str:
        return BigQueryLocation(**self.train_y, target_column=self.target_column, id_column=self.row_id_column)

    def get_eval_x(self) -> str:
        return BigQueryLocation(**self.eval_x, id_column=self.row_id_column)

    def get_eval_y(self) -> str:
        return BigQueryLocation(**self.eval_y, target_column=self.target_column, id_column=self.row_id_column)

    def _save_predicted_data(self, data_uri: BigQueryLocation, partition: str):
        data_uri = data_uri.copy()
        extra_columns = []
        extra_columns.append(
            f'"{self.mlflow_experiment_name}" as mlflow_experiment_name')
        extra_columns.append(f'"{self.model_id}" as model_id')
        extra_columns.append(f'"{self.model_version}" as model_version')
        extra_columns.append(f'"{self.run_id}" as run_id')
        extra_columns.append(f'"{partition}" as data_partition')
        extra_columns.append(f'CURRENT_TIMESTAMP as load_date')
        data_uri.data_columns += extra_columns
        query = data_uri.get_select_query(include_id=True)
        destination = f'{self.predictions_dataset}.{self.mlflow_experiment_name}'
        if (job_config := self.save_data_job_config) is None:
            job_config = set_defaults_save_job_config(
                bigquery.job.QueryJobConfig())
            job_config.destination = destination

        job = self.client.query(
            query, job_config=job_config
        )
        self.job_pool.append(job)

    def save_predicted_train_data(self, data_uri: BigQueryLocation):
        self._save_predicted_data(
            data_uri,
            partition='train'
        )

    def save_predicted_eval_data(self, data_uri: BigQueryLocation):
        self._save_predicted_data(
            data_uri,
            partition='eval'
        )

    def end_run(self):
        for job in self.job_pool:
            exception = job.exception(timeout=9999999)
            if exception:
                raise RuntimeError(exception)
