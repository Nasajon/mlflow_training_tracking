from google.cloud import bigquery


class BigQueryMixin:
    def __init__(self):
        self.client = bigquery.Client()

    def load_dataframe_from_query(self, query, **kwargs):
        print(f"load_dataframe_from_query: {query}")
        query_job = self.client.query(query,
                                      **kwargs)
        exception = query_job.exception()
        if exception:
            raise RuntimeError(exception)
        return query_job.to_dataframe()

    def run_query_and_wait(self, query, **kwargs):
        print(f"run_query_and_wait: {query}")
        query_job = self.client.query(query,
                                      **kwargs)
        # Seeting a bf timeout. Trying to get rid of '504 Deadline Exceeded'
        exception = query_job.exception(timeout=9999999)
        if exception:
            raise RuntimeError(exception)
        return query_job
