
class BigQueryLocation:
    base_select_query = """
    SELECT {columns} FROM `{table}` ORDER BY {order}"""

    def __init__(self, data_columns, id_column, table, order, target_column=None, limit=None):
        data_columns = data_columns.copy()
        self.data_columns = data_columns
        self.target_column = target_column
        self.id_column = id_column
        self.table = table
        self.order = order
        self.limit = limit

    def get_select_query(self, include_id=False):
        columns = self.data_columns.copy()
        if self.target_column:
            columns.append(self.target_column)
        if include_id:
            columns.append(self.id_column)
        select_query = self.base_select_query.format(
            columns=', '.join(columns),
            table=self.table,
            order=self.order)
        if self.limit:
            select_query = f"{select_query} LIMIT {self.limit }"
        return select_query

    def __repr__(self):
        repr = {
            'data_columns': self.data_columns,
            'id_column': self.id_column,
            'table': self.table,
            'order': self.order,
            'target_column': self.target_column,
            'limit': self.limit
        }
        return str(repr)

    def copy(self):
        return BigQueryLocation(
            data_columns=self.data_columns.copy(), id_column=self.id_column, table=self.table,
            order=self.order, target_column=self.target_column, limit=self.limit
        )
