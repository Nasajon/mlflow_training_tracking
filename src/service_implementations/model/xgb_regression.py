import xgboost as xgb
import os
from service_interfaces.model_interface import ModelOperatorInterface


class XGBRegressionModelOperatorDataFrame(xgb.XGBRegressor, ModelOperatorInterface):
    model_type = "XGB Tree Regression"
    __repr__ = object.__repr__

    def __init__(self):
        pass

    def instantiate_model(self, model_id, model_version, *args, **kwargs):
        self.model_id = model_id
        self.model_version = model_version
        self.model_id_version = f'{self.model_id}_{self.model_version}'
        super().__init__(**kwargs)

    def load(self, folder_path):
        path = os.path.join(folder_path, self.model_id_version)
        super().load_model(path)

    def save(self, folder_path):
        path = os.path.join(folder_path, self.model_id_version)
        super().save_model(path)

    def fit(self, train_x, train_y, eval_x, eval_y, **kwargs):
        eval_set_list = [(train_x, train_y, 'train'),
                         (eval_x, eval_y, 'eval')]
        self.eval_set_names = {f'validation_{index}': f"validation_{eval_set[2]}"
                               for index, eval_set in enumerate(eval_set_list)}
        # print(self.eval_set_names)
        kwargs['eval_set'] = [(eval_set[0], eval_set[1])
                              for eval_set in eval_set_list]
        super().fit(X=train_x, y=train_y, **kwargs)

    def get_train_metrics(self) -> 'list[dict]':
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
        metrics_struct_list = []
        aux_dict = {}
        values_list_len = 0
        for validation, metrics_collection in self.evals_result().items():
            for metric, values_list in metrics_collection.items():
                aux_dict[self.eval_set_names[validation] +
                         "."+metric] = values_list
                values_list_len = len(values_list)

        for index in range(values_list_len):
            metrics = {}
            for key in aux_dict.keys():
                metrics[key] = aux_dict[key][index]

            metrics_struct = {'metrics': metrics, 'step': index}
            metrics_struct_list.append(metrics_struct)
        return metrics_struct_list

    def feature_importance(self):
        return {}
