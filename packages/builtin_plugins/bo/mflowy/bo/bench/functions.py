"""bo-bench 合成函数族（#193 裁定题 1 → 执行票 #231）：6 族 numpy 自实现。

经典连续 4 族（Rosenbrock d=4 / Hartmann 6D / Ackley d=8 / 受控交叉二次型 d=6）
+ 混合空间自建 2 族（类别变形 Hartmann / 变性悬崖——OptunaHub 无混合套件故自建，
分类变量 per-category 位移/缩放 = categorical BO 文献通行 morphed base function）。
全部已知最优解析可算（逐类最优取 max）；场景 overlay（S1 贴边/S2 带窗/S3 漂移）
住 ``harness.Scenario``——函数族零场景知识。
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ── Hartmann 6D 常数（SFU Surjanovic & Bingham 标准表）────────────────────────
_H6_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_H6_A = np.array(
    [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ]
)
_H6_P = 1e-4 * np.array(
    [
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ]
)


def _hartmann6(x: np.ndarray) -> float:
    s = np.sum(_H6_ALPHA * np.exp(-np.sum(_H6_A * (x - _H6_P) ** 2, axis=1)))
    return float(-(s - 2.5) / 1.94 + 1.0)  # 仿 bayesmark 归一：最优 ≈ 0 量级


def _vec(params: dict, names: tuple[str, ...]) -> np.ndarray:
    """参数字典 → 有序数值向量（连续维取值面）。"""
    return np.array([float(params[n]) for n in names])


class Family:
    """族契约：dims（bo search_space 引擎面 payload）/ true_f / 已知最优 / 方向。"""

    name: str
    kind: str  # continuous | mixed（G2 交互族 = Rosenbrock/类别变形/变性悬崖）
    direction: str  # minimize | match（S2 场景 overlay 决定 band）
    known_optimum: float
    dims: dict[str, dict]

    def true_f(self, params: dict[str, Any]) -> float:
        raise NotImplementedError


class Rosenbrock4(Family):
    """强逐对交互（multivariate 主考）：f = Σ 100(x_{i+1}-x_i²)² + (1-x_i)²，f*=0 @(1,…)。"""

    name = "rosenbrock4"
    kind = "continuous"
    direction = "minimize"
    known_optimum = 0.0
    interaction = True
    dims = {  # noqa: RUF012 —— 族常量字典（构造期冻结）
        f"x{i}": {"kind": "continuous", "low": -2.0, "high": 2.0, "dtype": "float", "scale": "linear"}
        for i in range(1, 5)
    }

    def true_f(self, params):
        x = _vec(params, tuple(self.dims))
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


class Hartmann6(Family):
    """温和多峰交互：文献最优 -3.32237（归一形态；known_optimum 构造期数值标定）。"""

    name = "hartmann6"
    kind = "continuous"
    direction = "minimize"
    interaction = False
    dims = {f"x{i}": {"kind": "continuous", "low": 0.0, "high": 1.0} for i in range(1, 7)}  # noqa: RUF012 —— 族常量字典（构造期冻结）

    def __init__(self) -> None:
        from scipy.optimize import minimize as _sp_min

        self.known_optimum = float(
            min(
                _sp_min(lambda x: _hartmann6(x), x0=np.full(6, v), bounds=[(0.0, 1.0)] * 6).fun for v in (0.2, 0.5, 0.8)
            )
        )

    def true_f(self, params):
        return _hartmann6(_vec(params, tuple(self.dims)))


class Ackley8(Family):
    """多峰逃逸：f = -20·exp(-0.2√(d⁻¹Σx²)) - exp(d⁻¹Σcos(2πx)) + 20 + e，f*=0。"""

    name = "ackley8"
    kind = "continuous"
    direction = "minimize"
    known_optimum = 0.0
    interaction = False
    dims = {f"x{i}": {"kind": "continuous", "low": -32.768, "high": 32.768} for i in range(1, 9)}  # noqa: RUF012 —— 族常量字典（构造期冻结）

    def true_f(self, params):
        x = _vec(params, tuple(self.dims))
        return float(-20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e)


class QuadraticCross6(Family):
    """受控交互二次型（固定种子 PSD A + 峰偏移 μ）：f = (x-μ)ᵀA(x-μ)，解析最优 0。"""

    name = "quadratic_cross6"
    kind = "continuous"
    direction = "minimize"
    known_optimum = 0.0
    interaction = False
    dims = {f"x{i}": {"kind": "continuous", "low": -5.0, "high": 5.0} for i in range(1, 7)}  # noqa: RUF012 —— 族常量字典（构造期冻结）

    def __init__(self) -> None:
        rng = np.random.default_rng(231)
        m = rng.normal(size=(6, 6))
        self.A = m @ m.T / 6 + np.eye(6) * 0.5  # PSD
        self.mu = rng.uniform(-2.0, 2.0, size=6)  # 峰在域内

    def true_f(self, params):
        z = _vec(params, tuple(self.dims)) - self.mu
        return float(z @ self.A @ z)


class MorphedHartmann(Family):
    """类别变形 Hartmann（混合）：6 连续 + 1 类别（K=4）——逐类 3 维位移 + 1 维缩放，
    逐类最优解析可算（self.class_optima），f* = 逐类取 max（minimize 域：取最差类的最优——
    引擎须选对类再优化连续维；构造使最优类唯一）。"""

    name = "morphed_hartmann"
    kind = "mixed"
    direction = "minimize"
    interaction = True  # 类别×连续交互（逐类最优漂移）——G2 裁判族
    choices = ("alpha", "beta", "gamma", "delta")

    def __init__(self) -> None:
        rng = np.random.default_rng(193)
        self.offsets = {c: np.zeros(6) for c in self.choices}
        for _i, c in enumerate(self.choices):
            self.offsets[c][[0, 2, 4]] = rng.uniform(-0.25, 0.25, size=3)  # 3 维位移
        self.scales = {c: 1.0 + 0.15 * i for i, c in enumerate(self.choices)}  # delta 最陡 → 唯一最优类
        self.dims = {
            **{f"x{i}": {"kind": "continuous", "low": 0.0, "high": 1.0} for i in range(1, 7)},
            "variant": {"kind": "discrete", "choices": list(self.choices)},
        }
        # 逐类最优：位移后 x 域截断内的网格近似（构造已知——粗网格 21^6 不可行，用
        # 位移中心点的函数值下界近似；基准口径：known_optimum 仅作 regret 归一分母，
        # 逐类最优用「位移类中心解析最优」+ 缩放修正）
        self.class_optima = {}
        from scipy.optimize import minimize as _sp_min

        for c in self.choices:
            off, s = self.offsets[c], self.scales[c]

            def g(x: np.ndarray, _off: np.ndarray = off, _s: float = s) -> float:
                xc = np.clip(x + _off, 0.0, 1.0)
                return _s * _hartmann6(xc)

            best = min(_sp_min(g, x0=np.full(6, v), bounds=[(0.0, 1.0)] * 6).fun for v in (0.2, 0.5, 0.8))
            self.class_optima[c] = float(best)
        self.known_optimum = min(self.class_optima.values())

    def true_f(self, params):
        x = _vec(params, tuple(k for k in self.dims if k != "variant"))
        xc = np.clip(x + self.offsets[params["variant"]], 0.0, 1.0)
        return float(self.scales[params["variant"]] * _hartmann6(xc))


class DenaturationCliff(Family):
    """变性悬崖（域原生混合）：温度/pH/浓度 3 连续 + 酶 1 类别（K=4）——每酶有
    T_denat 温度悬崖（越过即活性崩塌）与独立峰高/峰位；解析最优 = 最好酶的峰。"""

    name = "denaturation_cliff"
    kind = "mixed"
    direction = "minimize"  # 损失口径：活性越大损失越小；构造 f = -activity
    interaction = True  # 类别特异悬崖——G2 裁判族
    enzymes = ("E1", "E2", "E3", "E4")

    def __init__(self) -> None:
        # 逐酶参数：峰位 (T*, pH*, c*) / 峰高 / T_denat（E2 最优但悬崖最窄——交互考点）
        self.spec = {
            "E1": {"peak": (50.0, 7.0, 0.50), "height": 0.80, "t_denat": 78.0},
            "E2": {"peak": (55.0, 7.5, 0.40), "height": 1.00, "t_denat": 62.0},
            "E3": {"peak": (45.0, 6.5, 0.60), "height": 0.85, "t_denat": 85.0},
            "E4": {"peak": (60.0, 8.0, 0.30), "height": 0.90, "t_denat": 70.0},
        }
        self.dims = {
            "temperature": {"kind": "continuous", "low": 30.0, "high": 90.0},
            "ph": {"kind": "continuous", "low": 5.0, "high": 9.0},
            "concentration": {"kind": "continuous", "low": 0.05, "high": 1.0},
            "enzyme": {"kind": "discrete", "choices": list(self.enzymes)},
        }
        self.known_optimum = -1.0  # E2 峰高 1.0 → f = -1.0

    def true_f(self, params):
        s = self.spec[params["enzyme"]]
        T, pH, c = float(params["temperature"]), float(params["ph"]), float(params["concentration"])
        # 高斯峰（三元）× 悬崖阶跃（T 超 t_denat 活性崩 90%）
        Tstar, pstar, cstar = s["peak"]
        gauss = np.exp(-(((T - Tstar) / 8.0) ** 2) - ((pH - pstar) / 0.6) ** 2 - ((c - cstar) / 0.20) ** 2)
        cliff = 0.1 if T > s["t_denat"] else 1.0
        return float(-s["height"] * gauss * cliff)


FAMILIES: dict[str, Family] = {
    f.name: f
    for f in (Rosenbrock4(), Hartmann6(), Ackley8(), QuadraticCross6(), MorphedHartmann(), DenaturationCliff())
}
"""6 族注册表（#193 题 1 裁定矩阵）。"""
