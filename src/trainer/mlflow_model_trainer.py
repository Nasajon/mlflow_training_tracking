import mlflow
import os
from service_interfaces.data_interface import DataOperatorInterface
from service_interfaces.model_interface import ModelOperatorInterface
from service_interfaces.evaluation_metrics_interface import EvaluationMetricsOperatorInterface


def prepend_key(data_obj: object, prepend: str):
    if isinstance(data_obj, dict):
        prepended_dict = {}
        for key in data_obj.keys():
            prepended_dict[prepend+str(key)] = data_obj.get(key)
        return prepended_dict

    if isinstance(data_obj, str):
        return prepend+data_obj

    if isinstance(data_obj, list):
        return [prepend_key(list_obj, prepend) for list_obj in data_obj]


class MachineLearningModelTrainer:

    def __init__(self,
                 mlflow_server,
                 mlflow_experiment_name,
                 model_id,
                 model_version,
                 model_interface, data_interface, metrics_interface,
                 train_data, eval_data, target_column, model_parameters, training_parameters,
                 custom_eval_metrics=[],
                 custom_train_metrics=[],
                 run_folder_path='/tmp/model_run',
                 model_folder='model_artifact',
                 log_folder='log',
                 error_log_folder='log',
                 model_tags={},
                 mlflow_logging_enabled=True):

        if not isinstance(model_interface, ModelOperatorInterface):
            raise TypeError
        if not isinstance(data_interface, DataOperatorInterface):
            raise TypeError
        if not isinstance(metrics_interface, EvaluationMetricsOperatorInterface):
            raise TypeError

        self.run_id = self.setup_mlflow(mlflow_server,
                                        mlflow_experiment_name)
        self.crate_folder_structure(run_folder_path,
                                    model_folder,
                                    log_folder,
                                    error_log_folder)
        self.log_name = 'model_trainer.log'
        self.error_log_name = 'model_trainer_error.log'
        self.log_path = os.path.join(self.log_folder_path,
                                     self.log_name)
        self.error_log_path = os.path.join(self.log_folder_path,
                                           self.error_log_name)
        self.model_id = model_id
        self.model_version = model_version
        self.model_interface = model_interface
        self.data_interface = data_interface
        self.metrics_interface = metrics_interface
        self.train_data = train_data
        self.eval_data = eval_data
        self.model_parameters = model_parameters
        self.training_parameters = training_parameters
        self.target_column = target_column
        self.custom_eval_metrics = custom_eval_metrics
        self.custom_train_metrics = custom_train_metrics
        self.model_tags = model_tags
        self.mlflow_logging_enabled = mlflow_logging_enabled

    def setup_mlflow(self, mlflow_server, mlflow_experiment_name):
        mlflow.set_tracking_uri(mlflow_server)
        mlflow.set_experiment(mlflow_experiment_name)
        run = mlflow.start_run()
        run_id = run.info.run_id

        return run_id

    def crate_folder_structure(self, run_folder_path,
                               model_folder,
                               log_folder,
                               error_log_folder):

        self.run_folder_path = os.path.join(run_folder_path, self.run_id)
        if os.path.exists(self.run_folder_path):
            raise RuntimeError("Run ID exist. Exiting")
        self.model_folder_path = os.path.join(run_folder_path, self.run_id,
                                              model_folder)
        self.log_folder_path = os.path.join(run_folder_path, self.run_id,
                                            log_folder)
        self.error_log_folder_path = os.path.join(run_folder_path, self.run_id,
                                                  error_log_folder)
        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)
        if not os.path.exists(self.log_folder_path):
            os.makedirs(self.log_folder_path)
        if not os.path.exists(self.error_log_folder_path):
            os.makedirs(self.error_log_folder_path)

    def load_data(self):
        print("Loading Data...")
        self.data_interface.load_data(
            self.train_data, self.eval_data, target_column=self.target_column)
        self.train_x = self.data_interface.get_train_x()
        self.train_y = self.data_interface.get_train_y()
        self.eval_x = self.data_interface.get_eval_x()
        self.eval_y = self.data_interface.get_eval_y()
        print("Data loaded.")
        return self

    def instantiate_model(self):
        self.model_interface.instantiate_model(
            self.model_id, self.model_version, **self.model_parameters)
        return self

    def print_kw(self, **kwargs):
        print(kwargs)

    def train(self):
        print("Training Model..")
        self.model_interface.fit(
            train_x=self.train_x,
            train_y=self.train_y,
            eval_x=self.eval_x,
            eval_y=self.eval_y,
            **self.training_parameters)
        print("Model trained.")
        return self

    def save_model(self):
        self.model_interface.save(self.model_folder_path)

    def get_eval_metrics(self):
        y_pred = self.predict(self.eval_x)
        self.metrics_interface.load_data(y_true=self.eval_y, y_pred=y_pred)
        return self.metrics_interface.get_eval_metrics()

    def get_train_metrics(self):
        return self.model_interface.get_train_metrics()

    def predict(self, data):
        predictions = self.model_interface.predict(data)
        return predictions

    def log(self, _type, log_obj, prepend=''):

        if isinstance(log_obj, list):
            for list_obj in log_obj:
                self.log(_type, list_obj, prepend)
            return

        if not self.mlflow_logging_enabled:
            print(log_obj)
            return
        if _type == 'params':
            log_obj = prepend_key(log_obj, prepend)
            mlflow.log_params(log_obj)

        if _type == 'metrics':
            log_obj['metrics'] = prepend_key(log_obj['metrics'], prepend)
            mlflow.log_metrics(**log_obj)

        if _type == 'artifact':
            mlflow.log_artifact(log_obj)

        if _type == 'artifacts':
            mlflow.log_artifacts(log_obj)

    def set_tags(self, **tags):
        if not self.mlflow_logging_enabled:
            print(tags)
            return
        mlflow.set_tags(tags)

    def pipeline(self):
        try:
            print('*'*20)
            print('*     Starting     *')
            print('*'*20)
            print(f'Model Run: {self.run_id}')
            self.set_tags(model_id=self.model_id,
                          version=self.model_version,
                          state='running',
                          model_type=self.model_interface.model_type,
                          **self.model_tags)
            print("Logging params...")
            self.log('params', {'train_data': str(self.train_data),
                                'eval_data': str(self.eval_data),
                                'target_column': str(self.target_column)})
            self.log('params', self.model_parameters, prepend='model.')
            self.log('params', self.training_parameters, prepend='training.')

            self.instantiate_model()
            self.load_data()
            self.train()

            print("Logging metrics..")
            self.log('metrics', self.get_train_metrics(), prepend='training.')
            self.log('metrics', self.get_eval_metrics(), prepend='eval.')
            for metric in self.custom_eval_metrics:
                metric_value = getattr(self.metrics_interface, metric)()
                self.log('metrics', metric_value, prepend='custom.')
            for metric in self.custom_train_metrics:
                metric_value = getattr(self.model_interface, metric)()
                self.log('metrics', metric_value, prepend='custom.')
            self.save_model()
            print("Logging artifacts...")
            self.log('artifacts', self.run_folder_path)
            self.set_tags(state='success')
            print("Run finished.")

        except Exception as e:
            self.set_tags(state='failed')
            print("Run Failed")
            import traceback
            print(traceback.format_exc())
            with open(self.error_log_path, 'w') as f:
                f.write(str(e))
                f.write(traceback.format_exc())
            self.log('artifact', self.error_log_path)
