from abc import ABCMeta, abstractmethod
from copy import deepcopy
from pandas import DataFrame


class ModelOperatorInterface(metaclass=ABCMeta):
    @property
    @abstractmethod
    def model_type(self):
        raise NotImplemented

    def __init__(self, _id, version, model_parameters, training_parameters, *args, **kwargs):
        self.id = _id
        self.version = version
        self.model_parameters = deepcopy(model_parameters)
        self.training_parameters = deepcopy(training_parameters)
        self.args = args
        self.kwargs = kwargs

    def get_parameters(self):
        parameters = {
            'id': self.id,
            'version': self.version,
            'target_column': self.target_column,
            'model_parameters': self.model_parameters,
            'training_parameters': self.training_parameters,
            **self.kwargs
        }
        if self.args:
            parameters['args'] = self.args
        return deepcopy(parameters)

    def get_model_parameters(self):
        return deepcopy(self.model_parameters)

    def get_training_parameters(self):
        return deepcopy(self.training_parameters)

    def setup(self, row_id_column, target_column, **kwargs):
        self.row_id_column = row_id_column
        self.target_column = target_column

    @abstractmethod
    def fit(train_x, train_y, eval_x, eval_y, *args, **kwargs) -> None:
        raise NotImplemented

    @abstractmethod
    def predict(self, data, *args, **kwargs) -> None:
        raise NotImplemented

    @abstractmethod
    def save(self, folder_path, *args, **kwargs) -> None:
        raise NotImplemented

    @abstractmethod
    def load(self, folder_path, *args, **kwargs) -> None:
        raise NotImplemented

    def save_misc_data(self):
        pass

    @abstractmethod
    def get_train_metrics(self, *args, **kwargs) -> 'list[dict]':
        """Return a list of dictionaries to log in metrics MLFlow server

        Returns:
            list[dict]: E.g.
            [
                {
                metrics: {
                    validadtion_1.metric_1: value,
                    validadtion_1.metric_2: value,
                    validadtion_2.metric_1: value,
                    validadtion_2.metric_2: value
                    }
                step: 1
                },
                                {
                metrics: {
                    validadtion_1.metric_1: value,
                    validadtion_1.metric_2: value,
                    validadtion_2.metric_1: value,
                    validadtion_2.metric_2: value
                    }
                step: 2
                }
            ]
        """
        raise NotImplemented
