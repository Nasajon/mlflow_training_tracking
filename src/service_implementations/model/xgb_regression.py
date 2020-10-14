import os
import xgboost as xgb
import pandas as pd
from service_interfaces.model_interface import ModelOperatorInterface
from helpers.util import df_temporary_remove_column, df_permanently_remove_column


class XGBRegressionModelOperatorDataFrame(xgb.XGBRegressor, ModelOperatorInterface):
    model_type = "XGB Tree Regression"
    __repr__ = object.__repr__

    def __init__(self, _id, version, model_parameters, training_parameters):
        self.id = _id
        self.version = version
        self.model_id_version = f'{self.id}_{self.version}'
        self.model_parameters = model_parameters
        self.training_parameters = training_parameters
        xgb.XGBRegressor.__init__(self, **model_parameters)
        ModelOperatorInterface.__init__(self, _id=_id, version=version,
                                        model_parameters=model_parameters, training_parameters=training_parameters)

    def load(self, folder_path):
        path = os.path.join(folder_path, self.model_id_version)
        super().load_model(path)

    def save(self, folder_path):
        path = os.path.join(folder_path, self.model_id_version)
        super().save_model(path)

    @df_permanently_remove_column(df_variable='train_x', column_property_name='row_id_column')
    @df_permanently_remove_column(df_variable='train_y', column_property_name='row_id_column')
    @df_permanently_remove_column(df_variable='eval_x', column_property_name='row_id_column')
    @df_permanently_remove_column(df_variable='eval_y', column_property_name='row_id_column')
    def fit(self, train_x, train_y, eval_x, eval_y):
        training_parameters = self.get_training_parameters()
        eval_set_list = [(train_x, train_y, 'train'),
                         (eval_x, eval_y, 'eval')]
        self.eval_set_names = {f'validation_{index}': f"validation_{eval_set[2]}"
                               for index, eval_set in enumerate(eval_set_list)}
        # print(self.eval_set_names)
        training_parameters['eval_set'] = [(eval_set[0], eval_set[1])
                                           for eval_set in eval_set_list]
        super().fit(X=train_x, y=train_y, **training_parameters)

    @df_temporary_remove_column(df_variable='x_uri', column_property_name='row_id_column')
    def predict(self, x_uri):
        y_pred = super().predict(x_uri)
        y_pred = pd.DataFrame(
            y_pred, columns=[f'predicted_{self.target_column}'])
        return y_pred

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
                         "." + metric] = values_list
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
