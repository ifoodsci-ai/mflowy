"""stop_on_error 中间件

全局错误处理中间件，决定工作流在任务失败时是否继续执行。
- stop_on_error=True（默认）：记录错误日志，原样抛出异常，工作流终止
- stop_on_error=False：记录错误日志，返回 None，工作流继续
"""

import logging

from mflowy.driver.context import Context
from mflowy.driver.handler import Handler
from mflowy.utils.logging import is_verbose

logger = logging.getLogger(__name__)


def stop_on_error(task: Context, next: Handler):
    try:
        return next(task)
    except Exception as e:
        if task.conf.stop_on_error:
            logger.error(
                f"Task [{task.conf.type}.{task.conf.module}]{task.conf.name} failed: {e}", exc_info=is_verbose()
            )
            raise
        logger.warning(
            f"Task [{task.conf.type}.{task.conf.module}]{task.conf.name} failed and skipped: {e}", exc_info=is_verbose()
        )
        return e
