from service_implementations.model.bigquery_base import ModelOperatorBigQueryLocation


class BigQueryXGBRegressionModelOperatorBigQueryLocation(ModelOperatorBigQueryLocation):
    model_type = "BigQuery BOOSTED_TREE_REGRESSOR"

    def instantiate_model(self, model_id, model_version, **kwargs):
        kwargs["model_type"] = "BOOSTED_TREE_REGRESSOR"
        super().instantiate_model(model_id, model_version, **kwargs)

    def feature_importance(self) -> dict:
        metrics_struct_list = {}
        train_feature_importance_query = self.train_feature_importance_query.format(
            sql_model_path=self.sql_model_path)
        query_job = self.run_query_and_wait(train_feature_importance_query,
                                            job_id_prefix=self.job_id_prefix)
        rows = query_job.result()
        for row in rows:
            metrics_struct_list[f"{row.feature}.importance_weight"] = row.importance_weight
            metrics_struct_list[f"{row.feature}.importance_gain"] = row.importance_gain
            metrics_struct_list[f"{row.feature}.importance_cover"] = row.importance_cover
        return {"metrics": metrics_struct_list}
