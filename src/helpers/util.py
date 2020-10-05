
import functools


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
