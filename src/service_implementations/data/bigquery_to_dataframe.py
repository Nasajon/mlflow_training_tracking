from datetime import datetime
from google.cloud import bigquery
from pandas import DataFrame
from service_implementations.data.dataframe_base import ToDataFrameBase
from helpers.default_settings import set_defaults_save_job_config


class BigQueryToDataFrame(ToDataFrameBase):

    def __init__(self, train_uri: str, eval_uri: str, target_column: str,
                 row_id_column: str, predictions_dataset: str,
                 save_data_job_config=None):
        """BigQuery Data implementation. Query API and return result as pandas Dataframe

        Args:
            train_uri (str): SQL query
            eval_uri (str): SQL query
            target_column (str): Target column

        Returns:
            self
        """
        super().__init__(
            train_uri, eval_uri, target_column, row_id_column,
            predictions_dataset=predictions_dataset,
            save_data_job_config='custom' if save_data_job_config is not None else None
        )
        self.predictions_dataset = predictions_dataset
        self.save_data_job_config = save_data_job_config
        self.client = bigquery.Client()
        self.data_loaded = False
        self.job_pool = list()

    def load_data(self):
        # get data that was previous loaded
        if self.data_loaded:
            return

        train_df = self.client.query(
            self.train_uri).to_dataframe()  # API request
        eval_df = self.client.query(
            self.eval_uri).to_dataframe()  # API request
        super().load_data(train_df=train_df, eval_df=eval_df)
        self.data_loaded = True

    def _save_predicted_data(self, data_uri: DataFrame, partition: str):
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

    def save_predicted_train_data(self, data_uri: DataFrame):
        self._save_predicted_data(
            data_uri,
            partition='train'
        )

    def save_predicted_eval_data(self, data_uri: DataFrame):
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
