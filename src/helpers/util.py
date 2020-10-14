
import functools
import pandas as pd


def force_async(fn):
    """
    Turns a sync function to async function using asyncio
    """
    import asyncio

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()

        partial_func = functools.partial(fn, *args, **kwargs)
        return await loop.run_in_executor(None, partial_func)

    return wrapper


def df_temporary_remove_column(df_variable: str, column_property_name: str):
    """
    Decorator to temporary remove column from data frame and add it later
    """

    @functools.wraps(df_variable, column_property_name)
    def wrapper(fn):

        @functools.wraps(fn)
        def wrapper_fn(self, *args, **kwargs):
            column = getattr(self, column_property_name)
            df = kwargs[df_variable].copy()
            df_column = df.pop(column).to_frame()
            kwargs[df_variable] = df
            ret_df = fn(self, *args, **kwargs)
            ret = df_column.join(ret_df)
            return ret
        return wrapper_fn
    return wrapper


def df_permanently_remove_column(df_variable: str, column_property_name: str):
    """
    Decorator to permanently remove column from data frame
    """

    @functools.wraps(df_variable, column_property_name)
    def wrapper(fn):
        @functools.wraps(fn)
        def wrapper_fn(self, *args, **kwargs):
            column = getattr(self, column_property_name)
            kwargs[df_variable] = kwargs[df_variable].drop(
                columns=column, inplace=False)
            return fn(self, *args, **kwargs)
        return wrapper_fn
    return wrapper
