from __future__ import annotations

import inspect
import threading
from functools import wraps


def synchronized(lock=None):
    """使用指定锁保护函数，确保同一时刻只有一个线程执行。

    注意：必须带括号调用（``@synchronized()`` / ``@synchronized(lock)``）——
    它是装饰器工厂，裸 ``@synchronized`` 会把函数当 lock 传入，装饰静默失效。
    """
    lock = lock or threading.Lock()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def synchronized_once(lock=None):
    """使用指定锁保护函数，确保同一装饰器实例内只执行一次（双重检查锁）。

    注意：必须带括号调用（``@synchronized_once()``），理由同 ``synchronized``。
    """
    lock = lock or threading.Lock()
    done = False
    result = None

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal done, result
            # 快速路径：已执行过，直接返回结果（避免锁竞争）
            if not done:
                with lock:
                    # 双重检查，防止多个线程同时进入锁
                    if not done:
                        result = func(*args, **kwargs)
                        done = True
            return result

        return wrapper

    return decorator


def silence(except_func=None, finally_func=None):
    """捕获异常并交给 except_func(e)，finally_func 无论成败都执行。

    Args:
        except_func: 异常回调（参数为异常对象），不传则静默吞掉
        finally_func: 无参收尾回调（成功/失败都执行）

    同步与异步函数均可装饰（按 iscoroutinefunction 分派 wrapper，异步异常在 await 阶段捕获）。
    """

    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)  # 必须 await
                except Exception as e:
                    if except_func:
                        except_func(e)
                finally:
                    if finally_func:
                        finally_func()

            return async_wrapper
        else:

            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if except_func:
                        except_func(e)
                finally:
                    if finally_func:
                        finally_func()

            return wrapper

    return decorator
