from pandas import DataFrame
import pandas as pd
from service_implementations.data.dataframe_base import ToDataFrameBase


class FileToDataFrame(ToDataFrameBase):
    def __init__(self, train_uri: str, eval_uri: str, target_column: str, row_id_column: str):
        self.data_loaded = False
        super().__init__(train_uri, eval_uri, target_column, row_id_column)

    def load_data(self):
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
        train_df = pd.read_csv(self.train_uri)
        # eval file
        eval_df = pd.read_csv(self.eval_uri)
        super().load_data(train_df=train_df, eval_df=eval_df)
        self.data_loaded = True

    def save_predicted_train_data(self, data_uri):
        pass

    def save_predicted_eval_data(self, data_uri):
        pass

    def end_run(self):
        pass
