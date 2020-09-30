from abc import ABCMeta, abstractmethod


class EvaluationMetricsOperatorInterface(metaclass=ABCMeta):
    @abstractmethod
    def load_data(X, y, *args, **kwargs):
        raise NotImplemented

    @abstractmethod
    def explained_variance_score(self):
        raise NotImplemented

    @abstractmethod
    def mean_absolute_error(self):
        raise NotImplemented

    @abstractmethod
    def mean_squared_error(self):
        raise NotImplemented

    @abstractmethod
    def median_absolute_error(self):
        raise NotImplemented

    @abstractmethod
    def r2_score(self):
        raise NotImplemented

    @abstractmethod
    def max_error(self):
        raise NotImplemented
