from copy import deepcopy
from service_interfaces.data_interface import DataOperatorInterface
from helpers.bigquery_location import BigQueryLocation


class DataOperatorBigQueryLocation(DataOperatorInterface):

    def __init__(self, train_uri: str, eval_uri: str, target_column: str, row_id_column: str):
        super().__init__(train_uri, eval_uri, target_column, row_id_column)

    def load_data(self) -> DataOperatorInterface:
        train_data = deepcopy(self.train_uri)
        eval_data = deepcopy(self.eval_uri)
        train_data['columns'].remove(self.target_column)
        self.train_y = train_data.copy()
        self.train_y.update({'columns': [self.target_column]})
        self.train_x = train_data.copy()
        self.train_x.update({'columns': train_data['columns']})
        eval_data['columns'].remove(self.target_column)
        self.eval_y = eval_data.copy()
        self.eval_y.update({'columns': [self.target_column]})
        self.eval_x = eval_data.copy()
        self.eval_x.update({'columns': eval_data['columns']})

        return self

    def get_train_x(self) -> str:
        return BigQueryLocation(**self.train_x)

    def get_train_y(self) -> str:
        return BigQueryLocation(**self.train_y)

    def get_eval_x(self) -> str:
        return BigQueryLocation(**self.eval_x)

    def get_eval_y(self) -> str:
        return BigQueryLocation(**self.eval_y)

    def save_predicted_train_data(self, data_uri):
        pass

    def save_predicted_eval_data(self, data_uri):
        pass

    def end_run(self):
        pass
