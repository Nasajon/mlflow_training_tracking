from google.cloud import bigquery
from pandas import DataFrame
from service_interfaces.data_interface import DataOperatorInterface


class BigQueryToDataFrame(DataOperatorInterface):
    data_loaded = False

    def __init__(self):
        self.client = bigquery.Client()

    def load_data(self, train_sql: str, eval_sql: str, target_column: str) -> DataOperatorInterface:
        """BigQuery Data implementation. Query API and return result as pandas Dataframe

        Args:
            train_sql (str): SQL query
            eval_sql (str): SQL query
            target_column (str): Target column

        Returns:
            self
        """
        # get data that was previous loaded
        if self.data_loaded:
            return

        df = self.client.query(train_sql).to_dataframe()  # API request
        self.train_y = df.pop(target_column)
        self.train_X = df
        df = self.client.query(eval_sql).to_dataframe()  # API request
        self.eval_y = df.pop(target_column)
        self.eval_X = df
        self.data_loaded = True
        return self

    def get_train_x(self) -> DataFrame:
        """Return X values as DataFrame

        Returns:
            DataFrame: X values from table
        """
        return self.train_X

    def get_train_y(self) -> DataFrame:
        """Return y values as DataFrame

        Returns:
            DataFrame: y values from table
        """
        return self.train_y

    def get_eval_x(self) -> DataFrame:
        """Return X values as DataFrame

        Returns:
            DataFrame: X values from table
        """
        return self.eval_X

    def get_eval_y(self) -> DataFrame:
        """Return y values as DataFrame

        Returns:
            DataFrame: y values from table
        """
        return self.eval_y
