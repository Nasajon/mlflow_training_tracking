from service_interfaces.data_interface import DataOperatorInterface
from helpers.bigquery_location import BigQueryLocation


class DataOperatorBigQueryLocation(DataOperatorInterface):

    def load_data(self, train_data: dict, eval_data: dict, target_column: str) -> DataOperatorInterface:
        train_data = train_data.copy()
        eval_data = eval_data.copy()
        train_data['columns'].remove(target_column)
        self.train_y = train_data.copy()
        self.train_y.update({'columns': [target_column]})
        self.train_x = train_data.copy()
        self.train_x.update({'columns': train_data['columns']})
        eval_data['columns'].remove(target_column)
        self.eval_y = eval_data.copy()
        self.eval_y.update({'columns': [target_column]})
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
