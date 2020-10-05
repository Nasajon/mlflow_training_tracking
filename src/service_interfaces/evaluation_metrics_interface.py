from abc import ABCMeta, abstractmethod


class EvaluationMetricsOperatorInterface(metaclass=ABCMeta):
    @abstractmethod
    def load_data(X, y, *args, **kwargs):
        raise NotImplemented

    @abstractmethod
    def get_eval_metrics(self, **kwargs):
        raise NotImplemented
