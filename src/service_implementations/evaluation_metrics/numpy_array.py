from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score, max_error
from sklearn.metrics._regression import _check_reg_targets
from service_interfaces.evaluation_metrics_interface import EvaluationMetricsOperatorInterface
import numpy as np


class EvaluationMetricsNumpyArray(EvaluationMetricsOperatorInterface):
    def load_data(self, y_true, y_pred, *args, **kwargs):
        y_type, y_true, y_pred, multioutput = _check_reg_targets(
            y_true, y_pred, multioutput='uniform_average')
        self.y_true = y_true
        self.y_pred = y_pred

    def explained_variance_score(self, **kwargs):
        return explained_variance_score(self.y_true, self.y_pred, **kwargs)

    def mean_absolute_error(self, **kwargs):
        return mean_absolute_error(self.y_true, self.y_pred, **kwargs)

    def mean_squared_error(self, **kwargs):
        return mean_squared_error(self.y_true, self.y_pred, **kwargs)

    def median_absolute_error(self, **kwargs):
        return median_absolute_error(self.y_true, self.y_pred, **kwargs)

    def r2_score(self, **kwargs):
        return r2_score(self.y_true, self.y_pred, **kwargs)

    def max_error(self, **kwargs):
        return max_error(self.y_true, self.y_pred, **kwargs)

    def mean_abs_perc_error(self, multioutput='uniform_average'):
        """Mean Absolute Percentage Error
        Calculate the mape.
        """
        return np.mean(np.abs((self.y_true - self.y_pred) / self.y_true))

    def percentile_absolute_error(self):
        abs_error = np.abs(self.y_true - self.y_pred)
        perc = np.percentile(abs_error, range(101), interpolation='nearest')
        metric_list = []
        for index in range(101):
            metric_list.append(
                {
                    "metrics": {"percentile_absolute_error": perc[index]}, 'step': index
                })
        return metric_list

    def get_eval_metrics(self, **kwargs):
        metrics = {}
        metrics["explained_variance_score"] = self.explained_variance_score()
        metrics["mean_absolute_error"] = self.mean_absolute_error()
        metrics["mean_squared_error"] = self.mean_squared_error()
        metrics["median_absolute_error"] = self.median_absolute_error()
        metrics["r2_score"] = self.r2_score()
        metrics["max_error"] = self.max_error()
        metrics["mean_abs_perc_error"] = self.mean_abs_perc_error()
        return {"metrics": metrics}
