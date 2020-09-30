
class BigQueryLocation:
    base_select_query = """
    SELECT {columns} FROM `{table}` ORDER BY {order}"""

    def __init__(self, columns, id_column, table, order, limit=None):
        self.columns = columns
        self.id_column = id_column
        self.table = table
        self.order = order
        self.limit = limit

    def get_select_query(self, include_id=False):
        select_query = self.base_select_query.format(
            columns=', '.join([self.id_column, *self.columns] if include_id
                              else self.columns),
            table=self.table,
            order=self.order)
        if self.limit:
            select_query = f"{select_query} LIMIT {self.limit }"
        return select_query

    def __repr__(self):
        repr = {
            'columns': self.columns,
            'id_column': self.id_column,
            'table': self.table,
            'order': self.order,
            'limit': self.limit
        }
        return str(repr)
