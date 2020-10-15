import mlflow
import os
import asyncio as aio
from helpers.util import force_async
from datetime import datetime
from service_interfaces.data_interface import DataOperatorInterface
from service_interfaces.model_interface import ModelOperatorInterface
from service_interfaces.evaluation_metrics_interface import EvaluationMetricsOperatorInterface
from helpers.exception_builder import ExceptionBuilder


def prepend_key(data_obj: object, prepend: str):
    if isinstance(data_obj, dict):
        prepended_dict = {}
        for key in data_obj.keys():
            prepended_dict[prepend + str(key)] = data_obj.get(key)
        return prepended_dict

    if isinstance(data_obj, str):
        return prepend + data_obj

    if isinstance(data_obj, list):
        return [prepend_key(list_obj, prepend) for list_obj in data_obj]


class MachineLearningModelTrainer:
    MAX_AIO_TASKS = 15

    def __init__(self,
                 mlflow_server,
                 mlflow_experiment_name,
                 model_interface, data_interface, metrics_interface,
                 custom_eval_metrics=[],
                 custom_train_metrics=[],
                 run_folder_path='/tmp/model_run',
                 model_folder='model_artifact',
                 log_folder='log',
                 error_log_folder='log',
                 experiment_parameters={},
                 experiment_tags={},
                 mlflow_param_logging_enabled=True,
                 mlflow_metric_logging_enabled=True,
                 mlflow_artifact_logging_enabled=True,
                 mlflow_logging_enabled=True):

        exception_builder = ExceptionBuilder(
            exception=TypeError, separator='; ')
        if not isinstance(model_interface, ModelOperatorInterface):
            exception_builder.add_message(
                "model_interface must be a instance of ModelOperatorInterface.")
        if not isinstance(data_interface, DataOperatorInterface):
            exception_builder.add_message(
                "data_interface must be a instance of DataOperatorInterface.")
        if not isinstance(metrics_interface, EvaluationMetricsOperatorInterface):
            exception_builder.add_message(
                "metrics_interface must be a instance of EvaluationMetricsOperatorInterface.")
        exception_builder.raise_exception_if_exist()

        self.model_interface = model_interface
        self.data_interface = data_interface
        self.metrics_interface = metrics_interface
        self.mlflow_experiment_name = mlflow_experiment_name

        self.run_id = self.setup_mlflow(mlflow_server,
                                        mlflow_experiment_name)
        self.create_folder_structure(run_folder_path,
                                     model_folder,
                                     log_folder,
                                     error_log_folder)
        self.log_name = 'model_trainer.log'
        self.error_log_name = 'model_trainer_error.log'
        self.log_path = os.path.join(self.log_folder_path,
                                     self.log_name)
        self.error_log_path = os.path.join(self.log_folder_path,
                                           self.error_log_name)

        self.custom_eval_metrics = custom_eval_metrics
        self.custom_train_metrics = custom_train_metrics
        self.experiment_tags = experiment_tags
        self.experiment_parameters = experiment_parameters
        self.mlflow_param_logging_enabled = mlflow_param_logging_enabled
        self.mlflow_metric_logging_enabled = mlflow_metric_logging_enabled
        self.mlflow_artifact_logging_enabled = mlflow_artifact_logging_enabled
        self.mlflow_logging_enabled = mlflow_logging_enabled

    @property
    def model_id(self):
        return self.model_interface.id

    @property
    def model_version(self):
        return self.model_interface.version

    @property
    def target_column(self):
        return self.data_interface.target_column

    @property
    def row_id_column(self):
        return self.data_interface.row_id_column

    def setup_mlflow(self, mlflow_server, mlflow_experiment_name):
        mlflow.set_tracking_uri(mlflow_server)
        mlflow.set_experiment(mlflow_experiment_name)
        run = mlflow.start_run(run_name=self.model_id)
        run_id = run.info.run_id

        return run_id

    def setup_interfaces(self):
        self.fprint('Interfaces Setup')
        setup = {
            'run_id': self.run_id,
            'model_id': self.model_id,
            'model_version': self.model_version,
            'mlflow_experiment_name': self.mlflow_experiment_name,
            'row_id_column': self.row_id_column,
            'target_column': self.target_column
        }
        self.data_interface.setup(**setup)
        self.model_interface.setup(**setup)
        self.metrics_interface.setup(**setup)

    async def end_run(self):
        if len(self.aio_tasks) > 0:
            self.fprint("Waiting async tasks")
            _done, self.aio_tasks = await aio.wait(self.aio_tasks, return_when=aio.ALL_COMPLETED)
            self.raise_task(_done)

        self._end_run()

    def _end_run(self):
        self.data_interface.end_run()
        mlflow.end_run()
        self.fprint("Run finished.")

    def create_folder_structure(self, run_folder_path,
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
        self.fprint('Loading Data...', end=' ')
        self.data_interface.load_data()
        self.train_x = self.data_interface.get_train_x()
        self.train_y = self.data_interface.get_train_y()
        self.eval_x = self.data_interface.get_eval_x()
        self.eval_y = self.data_interface.get_eval_y()
        self.fprint('Done.')
        return self

    def instantiate_model(self):
        self.fprint('Instantiating model...', end=' ')
        self.model_interface.instantiate_model(
            self.model_id, self.model_version, **self.model_parameters)
        self.fprint('Done.')
        return self

    def train(self):
        self.fprint('Training Model..')
        self.model_interface.fit(
            train_x=self.train_x,
            train_y=self.train_y,
            eval_x=self.eval_x,
            eval_y=self.eval_y)
        self.fprint('Model trained.')
        return self

    def save_model(self):
        if self.mlflow_artifact_logging_enabled:
            self.fprint('Saving model...', end=' ')
            self.model_interface.save(self.model_folder_path)
            self.fprint('Done.')

    def make_predictions(self):
        self.eval_y_pred = self.predict(self.eval_x)
        self.train_y_pred = self.predict(self.train_x)

    def save_predictions(self):
        self.data_interface.save_predicted_eval_data(self.eval_y_pred)
        self.data_interface.save_predicted_train_data(self.train_y_pred)

    def get_eval_metrics(self):
        self.metrics_interface.load_data(
            y_true=self.eval_y, y_pred=self.eval_y_pred)
        return self.metrics_interface.get_eval_metrics()

    def get_train_metrics(self):
        return self.model_interface.get_train_metrics()

    def predict(self, data):
        predictions = self.model_interface.predict(x_uri=data)
        return predictions

    def fprint(self, text, end='\n'):
        now = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S.%f")
        print(f'{now}- MFTT - {text}', end=end, flush=True)

    def raise_task(self, task_set):
        for task in task_set:
            if task.exception():
                raise task.exception()

    async def async_log(self, _type, log_obj, prepend=''):
        # self.fprint(f"Addeding log to async tasks.")

        if isinstance(log_obj, list):
            log_obj = log_obj.copy()
            for list_obj in log_obj:
                await self.async_log(_type, list_obj, prepend)
            return
        if isinstance(log_obj, dict):
            log_obj = log_obj.copy()

        if len(self.aio_tasks) >= self.MAX_AIO_TASKS:
            # self.fprint(f"Tasks set full, waiting any task to complete")
            _done, self.aio_tasks = await aio.wait(
                self.aio_tasks, return_when=aio.FIRST_COMPLETED)
            self.raise_task(_done)

        self.aio_tasks.add(aio.create_task(
            self._async_log(_type, log_obj, prepend)))

    @force_async
    def _async_log(self, _type, log_obj, prepend):
        self.__log(_type, log_obj, prepend)

    def __log(self, _type, log_obj, prepend=''):
        self.fprint(f"Logging {_type}...")

        if not self.mlflow_logging_enabled:
            print(log_obj)
            return

        if _type == 'params' and self.mlflow_param_logging_enabled:
            log_obj = prepend_key(log_obj, prepend)
            mlflow.log_params(log_obj)

        if _type == 'metrics' and self.mlflow_metric_logging_enabled:
            log_obj['metrics'] = prepend_key(log_obj['metrics'], prepend)
            mlflow.log_metrics(**log_obj)

        if _type == 'artifact' and self.mlflow_artifact_logging_enabled:
            mlflow.log_artifact(log_obj)

        if _type == 'artifacts' and self.mlflow_artifact_logging_enabled:
            mlflow.log_artifacts(log_obj)

    def set_tags(self, **tags):
        self.fprint('Setting tags...')
        if not self.mlflow_logging_enabled:
            print(tags)
            return
        mlflow.set_tags(tags)
        # self.fprint('Done.')

    async def _pipeline(self):
        try:
            print('*' * 20)
            print('*     Starting     *')
            print('*' * 20)
            self.fprint(f'Model Run: {self.run_id}')
            self.set_tags(model_id=self.model_id,
                          version=self.model_version,
                          state='running',
                          model_type=self.model_interface.model_type,
                          **self.experiment_tags)
            await self.async_log('params', self.experiment_parameters, prepend='exp.')
            await self.async_log('params', self.data_interface.get_parameters(), prepend='data.')
            await self.async_log('params', self.model_interface.get_model_parameters(), prepend='model.')
            await self.async_log('params', self.model_interface.get_training_parameters(),
                                 prepend='training.')

            self.setup_interfaces()
            self.load_data()
            self.train()
            self.make_predictions()
            self.save_predictions()

            await self.async_log('metrics', self.get_train_metrics(),
                                 prepend='training.')
            await self.async_log('metrics', self.get_eval_metrics(), prepend='eval.')
            for metric in self.custom_eval_metrics:
                metric_value = getattr(self.metrics_interface, metric)()
                await self.async_log('metrics', metric_value, prepend='custom.')
            for metric in self.custom_train_metrics:
                metric_value = getattr(self.model_interface, metric)()
                await self.async_log('metrics', metric_value, prepend='custom.')
            self.save_model()
            await self.async_log('artifacts', self.run_folder_path)
            self.set_tags(state='success')

        except Exception as e:
            print('\n\n\n')
            self.fprint('Run Failed')
            self.set_tags(state='failed')
            import traceback
            print(traceback.format_exc())
            with open(self.error_log_path, 'w') as f:
                f.write(str(e))
                f.write(traceback.format_exc())
            await self.async_log('artifact', self.error_log_path)

        finally:
            await self.end_run()

    def pipeline(self):
        self.aio_tasks = set()
        aio.run(self._pipeline())
