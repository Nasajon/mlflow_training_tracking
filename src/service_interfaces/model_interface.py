from abc import ABCMeta, abstractmethod
from pandas import DataFrame


class ModelOperatorInterface(metaclass=ABCMeta):
    @property
    @abstractmethod
    def model_type(self):
        raise NotImplemented

    @abstractmethod
    def instantiate_model(self, model_id, model_version, *args, **kwargs) -> None:
        raise NotImplemented

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
