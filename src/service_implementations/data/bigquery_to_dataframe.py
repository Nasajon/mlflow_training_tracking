from google.cloud import bigquery
from pandas import DataFrame
from service_interfaces.data_interface import DataOperatorInterface


class BigQueryToDataFrame(DataOperatorInterface):
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
        # self.load_mock(target_column)
        # return self
        df = self.client.query(train_sql).to_dataframe()  # API request
        self.train_y = df.pop(target_column)
        self.train_X = df
        df = self.client.query(eval_sql).to_dataframe()  # API request
        self.eval_y = df.pop(target_column)
        self.eval_X = df
        return self

    def load_mock(self, target_column):
        data_train = {'X': [1, 2, 3, 4, 5],
                      target_column: [1, 2, 3, 4, 5]}
        data_test = {'X': [6, 7, 8, 9, 10],
                     target_column: [6, 7, 8, 9, 10]}
        df_train = pd.DataFrame(data_train, columns=['X', target_column])
        df_test = pd.DataFrame(data_test, columns=['X', target_column])
        self.train_y = df_train.pop(target_column)
        self.train_X = df_train
        self.eval_y = df_test.pop(target_column)
        self.eval_X = df_test

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
