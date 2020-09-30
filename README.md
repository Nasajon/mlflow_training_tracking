Software para treinar modelos de machine learning e fazer o tracking na plataforma MLflow.


Implementar fase 1 do pipeline para treinar modelos de machine learning e fazer o tracking do treinamentos e resultado dos testes no MLflow

A implementação usará três interfaces:
Interface de dado - Responsável por ter o dado na representação necessária
Interface de modelo - Recebe o dado da interface de dado e faz o treinamento do modelo
Interface de avaliação - Calcula as métricas de teste do modelo treinado
Os dados pertinentes de cada etapa são armazenados no servidor do MLflow assim como os artefatos gerados

As interfaces precisam se 'conversar' com o mesmo tipo de dado, sendo assim necessário implementar uma interface por modelo e por tipo de dado

Nessa primeira fase serão implementadas as seguintes interfaces:
Interface de dado - Arquivo para DataFrame, BigQuery para DataFrame, BigQuery Location (URI)
Interface de modelo - XGBoost Regression (executado localmente) e BigQuery XGBoost Regression (executado na plataforma bigquery/ia platform no gcp)
Interface de Avaliação - Avaliação de Numpy Array e Avaliação de BigQuery Location em NumPy Array

Nessa primeira fase apenas as métricas de regressão foram implementadas, são elas:
explained_variance_score
mean_absolute_error
mean_squared_error
median_absolute_error
r2_score
max_error
mean_abs_perc_error
percentile_absolute_error

Os dados de treinamento são específicos para cada modelo.

Os artefatos gerados nessa etapa são os logs de error e o modelo para a implementação de XGBoost local, a implementação de BigQuery XGBoost ainda não salva o modelo no repositório de artefatos, mas o mesmo pode ser exportado na plataforma, essa exportação será feita posteriormente, assim como os logs de execução.

Também são armazenados no servidor todos os parâmetros utilizados para o treinamento
