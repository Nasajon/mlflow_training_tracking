from service_interfaces.model_interface import ModelOperatorInterface
from tensorflow import keras
import os


class KerasRegressionModelOperatorDataFrame(keras.Sequential, ModelOperatorInterface):
    model_type = "Keras DNN Regression"

    def instantiate_model(self, model_id, model_version, *args, **kwargs) -> None:
        self.model_id = model_id
        self.model_version = model_version
        self.model_id_version = f'{self.model_id}_{self.model_version}'
        layers_rep = kwargs['layers']
        layers = [getattr(keras.layers, layer['type'])(*layer.get('args', []),
                                                       **layer.get('kwargs', {})) for layer in layers_rep]
        del kwargs['layers']
        super().__init__(layers=layers, name=self.model_id_version)
        super().compile(**kwargs)

    def fit(self, train_x, train_y, eval_x, eval_y, **kwargs) -> None:
        kwargs['validation_data'] = (eval_x, eval_y)
        super().fit(x=train_x, y=train_y, **kwargs)

    def save(self, folder_path) -> None:
        path = os.path.join(folder_path, self.model_id_version)
        super().save_weights(path)

    def load(self, folder_path) -> None:
        path = os.path.join(folder_path, self.model_id_version)
        super().load_weights(path)

    def get_train_metrics(self):
        return []
