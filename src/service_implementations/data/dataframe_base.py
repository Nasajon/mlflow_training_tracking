from pandas import DataFrame
import pandas as pd
from service_interfaces.data_interface import DataOperatorInterface


class ToDataFrameBase(DataOperatorInterface):
    def __init__(self, train_uri: str, eval_uri: str, target_column: str, row_id_column: str, **kwargs):
        super().__init__(train_uri, eval_uri, target_column, row_id_column, **kwargs)

    def load_data(self, train_df, eval_df) -> DataOperatorInterface:
        self.train_y = train_df[[self.row_id_column, self.target_column]]
        self.train_X = train_df.drop(columns=self.target_column)
        self.eval_y = eval_df[[self.row_id_column, self.target_column]]
        self.eval_X = eval_df.drop(columns=self.target_column)

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
