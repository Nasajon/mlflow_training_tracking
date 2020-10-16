from abc import ABCMeta, abstractmethod


class EvaluationMetricsOperatorInterface(metaclass=ABCMeta):

    def setup(self, row_id_column, **kwargs):
        self.row_id_column = row_id_column

    @abstractmethod
    def load_data(X, y, *args, **kwargs):
        raise NotImplemented

    @abstractmethod
    def get_eval_metrics(self, **kwargs):
        raise NotImplemented
