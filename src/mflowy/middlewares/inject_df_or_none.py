"""可选 df 注入中间件：上游无 LOAD/CLEAN 步时注入 None 而非抛错。

用于 search_input 这类「df 仅作参考、可缺」的 handler——
无 data 时跳过推断，靠 yaml columns 提供搜索空间。
"""

from mflowy.driver.context import Context, PreviousContextNotFoundError
from mflowy.driver.handler import Handler
from mflowy.middlewares.data_inject import GetDF


def inject_df_or_none(ctx: Context, next: Handler):
    try:
        df = GetDF(ctx)
    except PreviousContextNotFoundError:
        df = None
    return next(ctx, df=df)
