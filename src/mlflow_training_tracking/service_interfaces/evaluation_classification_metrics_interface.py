from abc import ABCMeta, abstractmethod
import math
from mlflow_training_tracking.service_interfaces.evaluation_metrics_interface import EvaluationMetricsOperatorInterface


class EvaluationBinaryClassificationMetricsOperatorInterface(
        EvaluationMetricsOperatorInterface, metaclass=ABCMeta):

    @abstractmethod
    def true_positive(self):
        raise NotImplemented

    @abstractmethod
    def true_negative(self):
        raise NotImplemented

    @abstractmethod
    def false_positive(self):
        raise NotImplemented

    @abstractmethod
    def false_negative(self):
        raise NotImplemented

    @abstractmethod
    def precision(self):
        """
        The precision is the ratio tp / (tp + fp) where tp is the number of true positives
        and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label 
        as positive a sample that is negative.
        """
        raise NotImplemented

    @abstractmethod
    def true_positive_rate(self):
        """
        The recall (true_positive_rate) is the ratio tp / (tp + fn) where tp is the number
        of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        """
        raise NotImplemented

    @abstractmethod
    def true_negative_rate(self):
        """
        Specificity (true_negative_rate) measures the proportion of negatives that are correctly identified
        """
        raise NotImplemented

    def matthews_correlation_coefficient(self):
        """"
        The Matthews correlation coefficient is used in machine learning as a measure of 
        the quality of binary (two-class) classifications. It takes into account true 
        and false positives and negatives and is generally regarded as a balanced measure
        which can be used even if the classes are of very different sizes.
        The MCC is in essence a correlation coefficient value between -1 and +1.
        A coefficient of +1 represents a perfect prediction, 0 an average random prediction
        and -1 an inverse prediction. 
        The statistic is also known as the phi coefficient.
        """
        tp = self.true_positive()
        fp = self.false_positive()
        tn = self.true_negative()
        fn = self.false_negative()
        x = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        return ((tp * tn) - (fp * fn)) / math.sqrt(x)

    def get_eval_metrics(self, **kwargs):
        metrics = {}
        metrics["true_positive"] = self.true_positive()
        metrics["true_negative"] = self.true_negative()
        metrics["false_positive"] = self.false_positive()
        metrics["false_negative"] = self.false_negative()
        metrics["precision"] = self.precision()
        metrics["true_positive_rate"] = self.true_positive_rate()
        metrics["true_negative_rate"] = self.true_negative_rate()
        metrics["matthews_corr_coef"] = self.matthews_correlation_coefficient()
        return {"metrics": metrics}
