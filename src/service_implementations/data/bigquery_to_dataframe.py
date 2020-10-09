from google.cloud import bigquery
from pandas import DataFrame
from service_interfaces.data_interface import DataOperatorInterface


class BigQueryToDataFrame(DataOperatorInterface):

    def __init__(self, train_uri: str, eval_uri: str, target_column: str):
        """BigQuery Data implementation. Query API and return result as pandas Dataframe

        Args:
            train_uri (str): SQL query
            eval_uri (str): SQL query
            target_column (str): Target column

        Returns:
            self
        """
        super().__init__(train_uri, eval_uri, target_column)
        self.client = bigquery.Client()
        self.data_loaded = False

    def load_data(self) -> DataOperatorInterface:
        # get data that was previous loaded
        if self.data_loaded:
            return

        df = self.client.query(self.train_uri).to_dataframe()  # API request
        self.train_y = df.pop(self.target_column)
        self.train_X = df
        df = self.client.query(self.eval_uri).to_dataframe()  # API request
        self.eval_y = df.pop(self.target_column)
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
