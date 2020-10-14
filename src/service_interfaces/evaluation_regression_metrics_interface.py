from abc import ABCMeta, abstractmethod
from service_interfaces.evaluation_metrics_interface import EvaluationMetricsOperatorInterface


class EvaluationRegressionsMetricsOperatorInterface(EvaluationMetricsOperatorInterface, metaclass=ABCMeta):

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

    def get_eval_metrics(self, **kwargs):
        metrics = {}
        metrics["explained_variance_score"] = self.explained_variance_score()
        metrics["mean_absolute_error"] = self.mean_absolute_error()
        metrics["mean_squared_error"] = self.mean_squared_error()
        metrics["median_absolute_error"] = self.median_absolute_error()
        metrics["r2_score"] = self.r2_score()
        metrics["max_error"] = self.max_error()
        return {"metrics": metrics}
