"""词表身份契约（#67）：Builder 编译缝——ns 返回契约判据用例。

三份最小流是体检实证过的失败面，修复后必须编译通过（编译即校验：
词表 miss / 撞名 / 必选边不可达 / 类型不兼容全在 build() 暴露）。
"""

from pathlib import Path

from mflowy.driver.builder import Builder
from mflowy.model.templates import EXPLANATION_TEMPLATE


def build_flow(keys: list[str], tmp_path: Path):
    """最小流编译：键控平铺节点 + 空配置体（Annotated 参数编译期不校验）。"""
    p = tmp_path / "dag.yaml"
    p.write_text("flow:\n" + "".join(f"  - {k}: {{}}\n" for k in keys), encoding="utf-8")
    return Builder(p).build()


def test_shap_chain_compiles(tmp_path):
    """shap_explanation 裸节点寻址：plot 消费方必选边同名可达（原 plot_shap_explanation 不可达）。"""
    build_flow(["load__csv", "x_y", "model__loader", "shap_explanation", "plot__shap_summary"], tmp_path)


def test_explanation_template_compiles():
    """explanation compute 工具活路径：模板渲染后可编译（修复前尾部 plot 消费必选边不可达必炸）。"""
    steps = (
        "- load__csv:\n    name: 数据\n"
        "- x_y:\n    params: {targets: y}\n"
        "- model__loader:\n    params: {run_id: r1}\n"
    )
    Builder(
        EXPLANATION_TEMPLATE,
        env={"name": "t", "description": "d", "modeling_steps": steps, "lowess_frac": 0.3},
    ).build()
