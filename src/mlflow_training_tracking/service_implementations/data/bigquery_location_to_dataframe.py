from copy import deepcopy
from datetime import datetime
from google.cloud import bigquery
from mlflow_training_tracking.service_implementations.data.to_dataframe_base import ToDataFrameBase
from mlflow_training_tracking.helpers.bigquery_location import BigQueryLocation
from mlflow_training_tracking.helpers.default_settings import set_defaults_save_job_config


class DataOperatorBigQueryLocationToDataFrame(ToDataFrameBase):
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
        self.data_loaded = False
        self.client = bigquery.Client()
        self.job_pool = list()

    def load_data(self):
        if self.data_loaded:
            return

        train_data = deepcopy(self.train_uri)
        eval_data = deepcopy(self.eval_uri)

        bq_loc_train = BigQueryLocation(
            **train_data,
            target_column=self.target_column,
            id_column=self.row_id_column
        ).get_select_query(include_id=True)
        bq_loc_eval = BigQueryLocation(
            **eval_data,
            target_column=self.target_column,
            id_column=self.row_id_column
        ).get_select_query(include_id=True)

        train_df = self.client.query(
            bq_loc_train).to_dataframe()  # API request
        eval_df = self.client.query(
            bq_loc_eval).to_dataframe()  # API request
        super().load_data(train_df=train_df, eval_df=eval_df)
        self.data_loaded = True

    def _save_predicted_data(self, data_uri: 'DataFrame', partition: str):
        df = data_uri.copy()
        df['mlflow_experiment_name'] = str(self.mlflow_experiment_name)
        df['model_id'] = str(self.model_id)
        df['model_version'] = str(self.model_version)
        df['run_id'] = str(self.run_id)
        df['data_partition'] = str(partition)
        insertDate = datetime.utcnow()
        df['load_date'] = insertDate
        destination = f'{self.predictions_dataset}.{self.mlflow_experiment_name}'
        if (job_config := self.save_data_job_config) is None:
            job_config = set_defaults_save_job_config(
                bigquery.job.LoadJobConfig())

        job = self.client.load_table_from_dataframe(
            df, destination, job_config=job_config
        )
        self.job_pool.append(job)

    def save_predicted_train_data(self, data_uri: 'DataFrame'):
        self._save_predicted_data(
            data_uri,
            partition='train'
        )

    def save_predicted_eval_data(self, data_uri: 'DataFrame'):
        self._save_predicted_data(
            data_uri,
            partition='eval'
        )

    def end_run(self):
        # wait jobs to finish
        for job in self.job_pool:
            exception = job.exception(timeout=9999999)
            if exception:
                raise RuntimeError(exception)
