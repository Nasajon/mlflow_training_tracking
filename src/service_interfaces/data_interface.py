from abc import ABCMeta, abstractmethod
from pandas import DataFrame


class DataOperatorInterface(metaclass=ABCMeta):
    @abstractmethod
    def load_data(self, query: str, target_column: str, *args, **kwargs) -> 'DataOperatorInterface':
        """Get data from the database and return as pandas DataFrame

        Args:
            sql (str): SQL Query

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
