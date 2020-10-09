from abc import ABCMeta, abstractmethod
from pandas import DataFrame


class DataOperatorInterface(metaclass=ABCMeta):
    def __init__(self, train_uri: str, eval_uri: str, target_column: str, *args, **kwargs):
        self.train_uri = train_uri
        self.eval_uri = eval_uri
        self.target_column = target_column
        self.args = args
        self.kwargs = kwargs

    def get_parameters(self):
        parameters = {
            'train_uri': self.train_uri,
            'eval_uri': self.eval_uri,
            'target_column': self.target_column,
            **self.kwargs
        }
        if self.args:
            parameters['args'] = self.args
        return parameters

    @abstractmethod
    def load_data(self, *args, **kwargs) -> 'DataOperatorInterface':
        """Get data from the database and return as pandas DataFrame

        Args:
            query (str): Data location Query

        Raises:
            NotImplemented: Abstract Base Class

        Returns:
            object: Object to be passed to model fit, eg.: pandas DataFrame
        """
        raise NotImplemented

    @abstractmethod
    def get_train_x(self) -> DataFrame:
        raise NotImplemented

    @abstractmethod
    def get_train_y(self) -> DataFrame:
        raise NotImplemented

    @abstractmethod
    def get_eval_x(self) -> DataFrame:
        raise NotImplemented

    @abstractmethod
    def get_eval_y(self) -> DataFrame:
        raise NotImplemented
