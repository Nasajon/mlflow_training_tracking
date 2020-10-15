from google.cloud import bigquery


def set_defaults_save_job_config(job_config):
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    job_config.create_disposition = bigquery.CreateDisposition.CREATE_IF_NEEDED
    job_config.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        # If not set [field], the table is partitioned by pseudo column ``_PARTITIONTIME``.
        field=None
    )
    return job_config



