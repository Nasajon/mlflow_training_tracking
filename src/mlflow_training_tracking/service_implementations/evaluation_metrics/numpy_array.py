from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score, max_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics._regression import _check_reg_targets
from mlflow_training_tracking.service_interfaces.evaluation_regression_metrics_interface import EvaluationRegressionsMetricsOperatorInterface
from mlflow_training_tracking.service_interfaces.evaluation_classification_metrics_interface import EvaluationBinaryClassificationMetricsOperatorInterface
import numpy as np
from mlflow_training_tracking.helpers.util import df_permanently_remove_column


class EvaluationRegressionMetricsNumpyArray(EvaluationRegressionsMetricsOperatorInterface):

    @df_permanently_remove_column(df_variable='y_true', column_property_name='row_id_column')
    @df_permanently_remove_column(df_variable='y_pred', column_property_name='row_id_column')
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


class EvaluationBinaryClassificationMetricsNumpyArray(EvaluationBinaryClassificationMetricsOperatorInterface):

    def __init__(self, probability_threshold):
        self.probability_threshold = probability_threshold
        self.confusion_matrix_loaded = False

    @df_permanently_remove_column(df_variable='y_true', column_property_name='row_id_column')
    @df_permanently_remove_column(df_variable='y_pred', column_property_name='row_id_column')
    def load_data(self, y_true, y_pred, *args, **kwargs):
        y_type, y_true, y_pred, multioutput = _check_reg_targets(
            y_true, y_pred, multioutput='uniform_average')
        self.y_true = y_true
        self.y_pred = y_pred

    @property
    def confusion_matrix(self):
        if not self.confusion_matrix_loaded:
            self._confusion_matrix = confusion_matrix(
                self.y_true, self.y_pred).ravel()
            self.confusion_matrix_loaded = True
        return self._confusion_matrix

    def true_negative(self):
        return self.confusion_matrix[0]

    def false_positive(self):
        return self.confusion_matrix[1]

    def false_negative(self):
        return self.confusion_matrix[2]

    def true_positive(self):
        return self.confusion_matrix[3]

    def precision(self):
        """
        The precision is the ratio tp / (tp + fp) where tp is the number of true positives
        and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label 
        as positive a sample that is negative.
        """
        return self.true_positive() / (self.true_positive() + self.false_positive())

    def true_positive_rate(self):
        """
        The recall (true_positive_rate) is the ratio tp / (tp + fn) where tp is the number
        of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        """
        return self.true_positive() / (self.true_positive() + self.false_negative())

    def true_negative_rate(self):
        """
        Specificity (true_negative_rate) measures the proportion of negatives that are correctly identified
        tn / (tn + fp)
        """
        return self.true_negative() / (self.true_negative() + self.false_positive())
