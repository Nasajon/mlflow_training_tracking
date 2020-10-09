import os
from copy import deepcopy
from tensorflow import keras
from service_interfaces.model_interface import ModelOperatorInterface


class KerasRegressionModelOperatorDataFrame(keras.Sequential, ModelOperatorInterface):
    model_type = "Keras DNN Regression"

    def __init__(self, _id, version, model_parameters: dict, training_parameters: dict):
        self.model_id_version = f'{_id}_{version}'
        keras.Sequential.__init__(self, name=self.model_id_version)
        ModelOperatorInterface.__init__(
            self,
            _id=_id,
            version=version,
            model_parameters=model_parameters,
            training_parameters=training_parameters
        )

        self.layers_rep = self.model_parameters['layers']
        layers = [getattr(keras.layers, layer['type'])(*layer.get('args', []),
                                                       **layer.get('kwargs', {})) for layer in self.layers_rep]
        for layer in layers:
            self.add(layer)

        model_parameters = super().get_model_parameters()
        del model_parameters['layers']

        super().compile(**model_parameters)

    def get_model_parameters(self):
        model_parameters = super().get_model_parameters()
        del model_parameters['layers']
        index = 0
        for layer_rep in self.layers_rep:
            model_parameters[f'layer_{index}'] = layer_rep
            index += 1
        return deepcopy(model_parameters)

    def fit(self, train_x, train_y, eval_x, eval_y) -> None:
        training_parameters = super().get_training_parameters()
        print(training_parameters)
        training_parameters['validation_data'] = (eval_x, eval_y)
        super().fit(x=train_x, y=train_y, **training_parameters)

    def save(self, folder_path) -> None:
        path = os.path.join(folder_path, self.model_id_version)
        super().save_weights(path)

    def load(self, folder_path) -> None:
        path = os.path.join(folder_path, self.model_id_version)
        super().load_weights(path)

    def get_train_metrics(self):
        return []
