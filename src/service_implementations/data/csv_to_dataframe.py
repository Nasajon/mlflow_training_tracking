from pandas import DataFrame
import pandas as pd
from service_interfaces.data_interface import DataOperatorInterface


class FileToDataFrame(DataOperatorInterface):
    def __init__(self):
        self.data_loaded = False

    def load_data(self, train_file: str, eval_file: str, target_column: str) -> DataOperatorInterface:
        """CSV implementation. Open file and return as pandas Dataframe

        Args:
            train_file (str): File path
            eval_file (str): File path
            target_column (str): Target column

        Returns:
            self
        """
        # get data that was previous loaded
        if self.data_loaded:
            return

        # train file
        df = pd.read_csv(train_file)
        self.train_y = df.pop(target_column)
        self.train_X = df
        # eval file
        df = pd.read_csv(eval_file)
        self.eval_y = df.pop(target_column)
        self.eval_X = df
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
