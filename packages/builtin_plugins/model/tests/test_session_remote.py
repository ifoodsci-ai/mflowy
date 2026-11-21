"""run_hpo remote 分支契约（#218/#225 裁定 1/4）：provider 委派的会话半边——

- 分派：LocalJobProvider → 进程内直调（行为不变，另见 test_modeling_session）；
  其他 provider → 渲染 plan 提交 + 阻塞消费（提交 → 拉日志 → zip → 解包 → 查询 →
  study 采用 + OOF 镜像本地 run），账本条目对齐 local 分支
- 受理面：JobOutcome 二居其一——finished（进程内/mock，LocalRunFace 本地寻址）与
  submitted（RemoteJobProvider + 假 client 模拟平台终态与 zip 往返）
- 稳态 job-id 链：``inherit_platform_job_id``（工作区恢复）+ plan 内
  ``inherit_study_run_id``（study 寻址）两层（entry.remote 链簿）
- study 采用：add_trial 增量并入会话正典库（append-only）；混合历史/链断裂/
  全 pin 退化形/缺 platform_experiment_id fail-loud
"""

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mflowy.model.job_provider import set_job_provider
from mflowy.model.session import ModelingSession

_K_RF = "RF|passthrough|passthrough"
CV_CARD = "cv: 3"


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    """mlflow seam 隔离（会话库 = tmp tracking；工件根随 cwd=tmp 不污染仓库）。"""
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/tracking.db")
    monkeypatch.chdir(tmp_path)
    from mflowy.utils.path import pin_task_dir, reset_task_dir

    token = pin_task_dir(str(tmp_path))
    yield tmp_path
    reset_task_dir(token)
    set_job_provider(None)


@pytest.fixture
def reg_csv(tmp_path) -> str:
    n = 60
    rng = np.random.default_rng(4)
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "target": 2.0 * x + 0.1 * rng.normal(size=n)})
    p = tmp_path / "reg.csv"
    df.to_csv(p, index=False)
    return str(p)


def _start(csv, max_steps=8):
    return ModelingSession().load_or_start(
        file_path=csv,
        cv_card=CV_CARD,
        card={"domain": "测试", "target": "target"},
        task="regression",
        metric="mae",
        max_steps=max_steps,
        candidates=[{"family": "RF", "baseline": True}],
    )


def _state(observation_json):
    o = json.loads(Path(observation_json).read_text(encoding="utf-8"))
    b = json.loads((Path(observation_json).parent / "board.json").read_text(encoding="utf-8"))
    return {"cursor": b["cursor"], "models": b["models"], "steps": o["validation"]["steps"]}


# ── 产品面替身：真实 RF 搜索产 study.db + OOF 工件（同名工件契约的 Authentic 面）──


def _build_search_workspace(ws: Path, *, tag_model_node: bool, n_rows: int = 60) -> None:
    """在 ws 下产一个「远程任务工作区」：mlflow.db + .mlruns（模型步 run 持
    study.db/oof_y_pred.parquet/半宽 tag——OOF 产面移进管线后的同名工件）。"""
    import mlflow
    from mflowy.data.types import TASKTYPE
    from mflowy.model.cv.split import split as split_fn
    from mflowy.model.model._split import materialize_split
    from mflowy.model.model.random_forest import RF
    from mflowy.model.model.types import MetricName
    from mflowy.model.x_y import XySpec

    ws.mkdir(parents=True, exist_ok=True)
    prev = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{ws / 'mlflow.db'}")
    try:
        mlflow.create_experiment("remote-ws", artifact_location=str(ws / ".mlruns"))
        exp_id = mlflow.get_experiment_by_name("remote-ws")
        assert exp_id is not None  # 刚创建（Optional 收窄）
        rng = np.random.default_rng(4)
        x = rng.normal(size=n_rows)
        df = pd.DataFrame({"x": x, "target": 2.0 * x + 0.1 * rng.normal(size=n_rows)})
        X, y = df[["x"]], df[["target"]]
        assert isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame)  # 列清单索引恒 DataFrame（联合噪声）
        xy = XySpec(X, y, TASKTYPE.REGRESSION)
        ts = materialize_split(split_fn(x_y=xy, cv=3), xy)
        from mflowy.model.model.study import reset_study_live, set_study_live

        token = set_study_live(True)  # LocalJobProvider 同款窗口：study.db 直驻 run 工件根
        try:
            with mlflow.start_run(experiment_id=exp_id.experiment_id, run_name="会话搜索-RF") as run:
                RF(ts, n_trials=2, scoring=MetricName.MAE)
                if tag_model_node:
                    mlflow.set_tag("mflowy.node", "model__RF")
        finally:
            reset_study_live(token)
    finally:
        mlflow.set_tracking_uri(prev)
    return run.info.run_id


def _continue_search_workspace(ws_src: Path, ws_dst: Path, anchor_run_id: str, *, n_trials: int = 2) -> None:
    """续跳工作区（继承链滚雪球模拟）：copy 前史树 → 新模型步 run 以
    ``inherit_study_run_id`` 锚定前史 study（#244 外部锚：trial 追加进锚树，本 run
    无自有 study.db——链锚定位的消费面实证）。"""
    import shutil

    import mlflow
    from mflowy.data.types import TASKTYPE
    from mflowy.model.cv.split import split as split_fn
    from mflowy.model.model._split import materialize_split
    from mflowy.model.model.random_forest import RF
    from mflowy.model.model.types import MetricName
    from mflowy.model.x_y import XySpec

    shutil.copytree(ws_src, ws_dst)
    prev = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{ws_dst / 'mlflow.db'}")
    try:
        rng = np.random.default_rng(4)
        x = rng.normal(size=60)
        df = pd.DataFrame({"x": x, "target": 2.0 * x + 0.1 * rng.normal(size=60)})
        X, y = df[["x"]], df[["target"]]
        assert isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame)
        xy = XySpec(X, y, TASKTYPE.REGRESSION)
        ts = materialize_split(split_fn(x_y=xy, cv=3), xy)
        exp_id = mlflow.get_experiment_by_name("remote-ws")
        assert exp_id is not None  # copytree 带入
        with mlflow.start_run(experiment_id=exp_id.experiment_id, run_name="会话搜索-RF#2"):
            RF(ts, n_trials=n_trials, scoring=MetricName.MAE, inherit_study_run_id=anchor_run_id)
            mlflow.set_tag("mflowy.node", "model__RF")
    finally:
        mlflow.set_tracking_uri(prev)


def _zip_dir(ws: Path, dest: Path) -> None:
    with zipfile.ZipFile(dest, "w") as zf:
        for p in sorted(ws.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(ws))


# ── 假平台面：RemoteJobProvider 的 client 工厂缝（真 provider + 假 transport）──


class FakePlatform:
    """平台假体：preset 解析 + create_job + 终态轮询 + outputs zip 供给。"""

    def __init__(self, zips: dict[str, Path | None]):
        self.zips = zips
        self.created: list[dict] = []

    def list_presets(self):
        from mflowy.model.job_provider.platform_client import JobPreset

        return [
            JobPreset(
                id="p1",
                name="cpu-test",
                version=1,
                cpu_millis=1000,
                memory_mi=1024,
                active_deadline_seconds=600,
                image="img",
                ttl_seconds_after_finished=60,
                workdir="/",
            )
        ]

    def list_files(self, experiment_id, mount_type, prefix=""):
        from mflowy.model.job_provider.platform_client import FileEntry

        return [
            FileEntry(mount_type="outputs", filename=f"{jid}.zip", size=1, etag="e", last_modified=1)
            for jid in self.zips
        ]

    def create_job(self, cmd, args, experiment_id, preset_id):
        from mflowy.model.job_provider.platform_client import Job

        self.created.append({"cmd": cmd, "args": list(args), "experiment_id": experiment_id})
        job_id = f"job-{len(self.created)}"
        self.zips.setdefault(job_id, None)  # 占位（dict[str, Path | None]——测试侧随后供 zip 源）
        return Job(id=job_id, cmd=cmd, args=list(args), preset_id=preset_id, status="succeeded")

    def get_job(self, job_id):
        from mflowy.model.job_provider.platform_client import Job

        return Job(id=job_id, cmd="mflowy-job", args=[], preset_id="p1", status="succeeded")

    def get_job_logs(self, job_id, tail_lines):
        return f"[fake] {job_id} 完成"

    def get_download_url(self, experiment_id, mount_type, filename):
        job_id = filename[: -len(".zip")]
        ws = self.zips[job_id]
        assert ws is not None, f"job {job_id} 无 zip 源供给"
        out = ws.parent / f"served-{filename}"
        if not out.exists():
            _zip_dir(ws, out)
        return f"file://{out}", None


def _install_remote_provider(monkeypatch, fake: FakePlatform):
    from mflowy.model.job_provider.remote import RemoteJobProvider

    monkeypatch.setenv("MFLOWY_PLATFORM_JOB_PRESET", "cpu-test")
    set_job_provider(RemoteJobProvider(client=fake))
    import shutil

    def _fake_download(url, dest, *, transport=None):
        """file:// 供给面（假平台无 HTTP——真 download_presigned 只走 presigned HTTP）。"""
        assert url.startswith("file://"), f"假平台只收 file://，got {url}"
        shutil.copyfile(url.removeprefix("file://"), dest)

    monkeypatch.setattr("mflowy.model.job_provider.platform_client.download_presigned", _fake_download)


def _pin_all_dynamic_dims(observation_json: str) -> None:
    """光标全 pin（动态维取空间起点）——remote 退化形门测试的前置。"""
    from mflowy.model import registry
    from mflowy.model.model.study import ContinuousSpace, DiscreteSpace

    face = registry.family_face("RF")
    pins = {
        k: v.start
        for k, v in face.items()
        if isinstance(v, (ContinuousSpace, DiscreteSpace)) and not isinstance(v, DiscreteSpace)
    }
    pins.update({k: next(iter(v)) for k, v in face.items() if isinstance(v, DiscreteSpace)})
    ModelingSession().apply(observation_json, "set_hp", pins)


# ── submitted 全链：真 RemoteJobProvider + 假平台（uploads plan / 链 / 采用 / 镜像）──


def test_run_hpo_submitted_full_chain(tmp_path, monkeypatch, isolate, reg_csv):
    """submitted 全链：plan uploads 寻址 + 提交 → 终态拉 zip → 解包查询 → study
    采用入正典库 + OOF 镜像本地 run——账本条目对齐 local 分支形状 + remote 链簿。"""
    observation_json, _ = _start(reg_csv)
    ModelingSession().apply(observation_json, "select_model", {"family": "RF"})

    import optuna

    fake = FakePlatform(zips={})
    _install_remote_provider(monkeypatch, fake)

    ws = tmp_path / "remote-ws-1"
    anchor = _build_search_workspace(ws, tag_model_node=True)
    assert isinstance(anchor, str) and anchor  # 树主 run_id（断跳链锚断言面）
    fake.zips["job-1"] = ws  # get_download_url 现场打包（zip 本体即弃）

    # 远程工件 URI 前缀改写实测：monkeypatch 常量单源 → ws 根（zip 契约模拟）
    import mflowy.utils.constants as consts

    monkeypatch.setattr(consts, "REMOTE_TASK_DIR", str(ws))

    r = ModelingSession().apply(observation_json, "run_hpo", {"n_trials": 2, "platform_experiment_id": "exp-1"})

    # plan 面：uploads 寻址 + cv.split 确定性复现 + search 行为参数
    args = fake.created[0]["args"]
    assert args[:3] == ["wrap", "--tool", "hyperparameters_search"]
    import yaml as _yaml

    payload = _yaml.safe_load(args[4])
    steps = _yaml.safe_load(payload["modeling_steps"])
    load_params = next(s["load__file"]["params"] for s in steps if "load__file" in s)
    assert load_params["file_path"] == f"/user-data/uploads/{Path(reg_csv).name}"
    assert any("cv__split" in s for s in steps), "remote 分支须 cv.split 确定性复现（无本地 run 引用可依）"
    model_params = next(s["model__RF"]["params"] for s in steps if "model__RF" in s)
    assert model_params["n_trials"] == 2 and model_params["scoring"] == "MAE"

    # 账本：对齐 local 分支（study/objective/oof/y_pred/half_width/u_inner/best_params）
    st = _state(observation_json)
    entry = st["models"][_K_RF]
    assert entry["study"] == "study_RF__passthrough__passthrough.db"
    from mflowy.utils.constants import RANDOM_STATE

    assert entry["objective"] == {"seed": RANDOM_STATE, "scoring": "mae"}  # 候选缺省 seed 随 select_model 注入
    assert entry["oof"]["y_pred"] == "oof_y_pred.parquet" and entry["chain"]["job_id"] == "job-1"
    assert entry["u_inner"] < 0 and entry["best_params"]  # MAE 误差类取负 + θ* 在场
    session_db = Path(observation_json).parent / entry["study"]
    study = optuna.load_study(study_name=_K_RF, storage=f"sqlite:///{session_db}")
    assert len(study.trials) == 2 and study.user_attrs["objective"] == {"seed": RANDOM_STATE, "scoring": "mae"}
    assert entry["n_trials_at_last_eval"] == 2 and entry["chain"]["n_adopted"] == 2

    # OOF 镜像 run：本地库可寻址（指纹 tag + 证据 parquet + study db 投影）
    import mlflow
    from mflowy.utils.mlflow import run_artifacts

    mirror = entry["oof"]["run_id"]
    arts = run_artifacts(mirror)
    assert "oof_y_pred.parquet" in arts and "study_RF__passthrough__passthrough.db" in arts
    run = mlflow.get_run(mirror)
    assert run.data.tags["mflowy.session_id"] and run.info.run_name == f"run_hpo:{_K_RF}"
    o = json.loads(Path(observation_json).read_text(encoding="utf-8"))
    assert run.data.tags["mflowy.source_fingerprint"] == o["biz"]["source_fingerprint"]  # 指纹 tag = 开局帧指纹
    assert r.validation.steps == 1 and r.biz["board"][_K_RF]["u_inner"] == entry["u_inner"]  # 板面投影随账本

    # 第二跳（稳态链）：inherit_platform_job_id=job-1 + plan 内 inherit_study_run_id；
    # ws2 = 前史树滚雪球 + 锚定续跑（锚树 study 4 trials、新 run 无自有 db）
    ws2 = tmp_path / "remote-ws-2"
    _continue_search_workspace(ws, ws2, anchor)
    fake.zips["job-2"] = ws2
    import mflowy.utils.constants as consts

    monkeypatch.setattr(consts, "REMOTE_TASK_DIR", str(ws2))
    ModelingSession().apply(observation_json, "run_hpo", {"n_trials": 2, "platform_experiment_id": "exp-1"})
    args2 = fake.created[1]["args"]
    assert "--inherit-from" in args2 and args2[args2.index("--inherit-from") + 1] == "job-1"
    payload2 = _yaml.safe_load(args2[4])
    model_params2 = next(
        s["model__RF"]["params"] for s in _yaml.safe_load(payload2["modeling_steps"]) if "model__RF" in s
    )
    assert model_params2["inherit_study_run_id"]  # 节点级 study 寻址（首跳模型步 run）
    entry2 = _state(observation_json)["models"][_K_RF]
    assert entry2["chain"]["job_id"] == "job-2" and entry2["chain"]["n_adopted"] == 4  # 同规模 → 增量 0


# ── finished outcome（进程内 provider / mock）：LocalRunFace 本地寻址 ──────────


def test_run_hpo_finished_outcome_consumes_dag_result(tmp_path, isolate, reg_csv):
    """finished 受理（自定义进程内 provider）：DAGResult 节点 run 本地寻址——同一
    套采用/镜像机械，mock/瘦装门（CI）即此形态。"""
    observation_json, _ = _start(reg_csv)
    ModelingSession().apply(observation_json, "select_model", {"family": "RF"})

    # 供体 run：真实 RF search（本库 tracking）——study/OOF 同名工件
    import mlflow
    from mflowy.data.types import TASKTYPE
    from mflowy.driver.context import NodeResult
    from mflowy.driver.dag import DAGResult
    from mflowy.model.cv.split import split as split_fn
    from mflowy.model.job_provider.protocol import JobOutcome
    from mflowy.model.model._split import materialize_split
    from mflowy.model.model.random_forest import RF
    from mflowy.model.model.types import MetricName
    from mflowy.model.x_y import XySpec

    rng = np.random.default_rng(4)
    x = rng.normal(size=60)
    df = pd.DataFrame({"x": x, "target": 2.0 * x + 0.1 * rng.normal(size=60)})
    X, y = df[["x"]], df[["target"]]
    assert isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame)  # 列清单索引恒 DataFrame（联合噪声）
    xy = XySpec(X, y, TASKTYPE.REGRESSION)
    ts = materialize_split(split_fn(x_y=xy, cv=3), xy)
    from mflowy.model.model.study import reset_study_live, set_study_live

    token = set_study_live(True)  # provider 同款窗口（study.db 直驻 run 工件根）
    try:
        with mlflow.start_run(run_name="donor:search") as run:
            RF(ts, n_trials=2, scoring=MetricName.MAE)
    finally:
        reset_study_live(token)
    donor = run.info.run_id

    class InProcessProvider:
        async def hyperparameters_search(self, *, modeling_steps, name, desc, **kw):
            return JobOutcome.finished(
                dag_result=DAGResult(
                    experiment_id="0",
                    status="finished",
                    error="",
                    graph="",
                    runs=[
                        NodeResult(
                            node="model__RF",
                            name="会话搜索-RF",
                            run_id=donor,
                            status="FINISHED",
                            artifacts=[],
                            output="",
                        )
                    ],
                )
            )

    set_job_provider(InProcessProvider())  # type: ignore[arg-type]  # 测试替身只实现受理面（协议其余方法不进本场）

    ModelingSession().apply(observation_json, "run_hpo", {"n_trials": 2, "platform_experiment_id": "exp-x"})
    entry = _state(observation_json)["models"][_K_RF]
    assert entry["chain"]["job_id"] is None and entry["chain"]["run_id"] == donor
    assert entry["chain"]["n_adopted"] == 2 and entry["u_inner"] < 0
    from mflowy.utils.mlflow import run_artifacts

    assert "oof_y_pred.parquet" in run_artifacts(entry["oof"]["run_id"])  # 镜像 run 在场


# ── fail-loud 门 ────────────────────────────────────────────────────────────


def test_run_hpo_remote_requires_platform_experiment_id(tmp_path, monkeypatch, isolate, reg_csv):
    observation_json, _ = _start(reg_csv)
    ModelingSession().apply(observation_json, "select_model", {"family": "RF"})
    _install_remote_provider(monkeypatch, FakePlatform(zips={}))
    with pytest.raises(ValueError, match="ensure_platform_experiment"):
        ModelingSession().apply(observation_json, "run_hpo", {"n_trials": 2})
    assert _state(observation_json)["models"] == {}  # 拒绝在提交前（失败吃步但零账本条目）


def test_run_hpo_degenerate_all_pin_via_single_value_space(tmp_path, isolate, reg_csv):
    """全 pin 退化形（有效空间空 × n_trials=1）经 provider 保真：单值 DiscreteSpace
    渲染保持 search 行为（inner CV 评估，不碰 holdout——DAG 两行为判据下裸全 pin
    会转 training 越权读 test）。"""
    observation_json, _ = _start(reg_csv)
    ModelingSession().apply(observation_json, "select_model", {"family": "RF"})
    _pin_all_dynamic_dims(observation_json)
    ModelingSession().apply(observation_json, "run_hpo", {"n_trials": 1})
    entry = _state(observation_json)["models"][_K_RF]
    assert entry["u_inner"] is not None and entry["chain"]["n_adopted"] == 1
    from mflowy.utils.mlflow import run_artifacts

    arts = run_artifacts(entry["oof"]["run_id"])
    assert "oof_y_pred.parquet" in arts and "X_test.parquet" not in arts  # search 面：无 holdout 副产物


def test_run_hpo_rejects_degenerate_with_extra_trials(tmp_path, isolate, reg_csv):
    """退化形 × n_trials>1 仍拒绝（多余 trial 是同一参数的重复评估）。"""
    observation_json, _ = _start(reg_csv)
    ModelingSession().apply(observation_json, "select_model", {"family": "RF"})
    _pin_all_dynamic_dims(observation_json)
    with pytest.raises(ValueError, match="n_trials=1 退化形"):
        ModelingSession().apply(observation_json, "run_hpo", {"n_trials": 2})


def test_run_hpo_rejects_legacy_history_outside_chain(tmp_path, monkeypatch, isolate, reg_csv):
    """链外历史门：provider 化之前的存量会话库（有 trial、无链簿/采用留痕）→
    拒绝续跑（红线 1 宁可拒绝——存量 study 语义不可经链重建）。"""
    observation_json, _ = _start(reg_csv, max_steps=8)
    ModelingSession().apply(observation_json, "select_model", {"family": "RF"})
    # 直接在正典位造存量 study（模拟 provider 化之前的会话：有 trial、零链面）
    import optuna
    from mflowy.model.session import _study_db_name

    db = Path(observation_json).parent / _study_db_name(_K_RF)
    legacy = optuna.create_study(study_name=_K_RF, storage=f"sqlite:///{db}", direction="minimize")
    legacy.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=2)
    legacy.set_user_attr("objective", {"seed": 42, "scoring": "mae"})
    _install_remote_provider(monkeypatch, FakePlatform(zips={}))
    with pytest.raises(RuntimeError, match="链外历史"):
        ModelingSession().apply(observation_json, "run_hpo", {"n_trials": 1, "platform_experiment_id": "exp-1"})


def test_await_job_failed_terminal_raises(tmp_path, isolate):
    """失败终态 → 动作失败（失败吃步由壳钉——见 driver 测试）：错误面带 job 与日志指引。"""
    from mflowy.model import session_remote
    from mflowy.model.job_provider.platform_client import Job

    class FailingClient:
        def get_job(self, job_id):
            return Job(id=job_id, cmd="mflowy-job", args=[], preset_id="p", status="failed", error_message="boom")

        def get_job_logs(self, job_id, tail_lines):
            return ""

    with pytest.raises(RuntimeError, match=r"失败终态.*boom"):
        session_remote.await_job(FailingClient(), "job-x", interval=0.01)


def test_adopt_study_trials_chain_break_fails_loud(tmp_path, isolate):
    """链断裂（远程 trial 计数 < 已采用）→ 拒绝静默缩水（红线 1）。"""
    import optuna
    from mflowy.model import session_remote

    remote_db = tmp_path / "remote.db"
    remote = optuna.create_study(study_name="study", storage=f"sqlite:///{remote_db}")
    remote.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=1)

    session_db = tmp_path / "session.db"
    with pytest.raises(RuntimeError, match="继承链断裂"):
        session_remote.adopt_study_trials(
            session_db,
            study_name="k",
            objective_id={"seed": "default", "scoring": "mae"},
            direction="minimize",
            seed=None,
            remote_db=remote_db,
            n_adopted=5,
            remote_study_name="study",
        )


def test_adopt_study_trials_incremental_and_objective_gate(tmp_path, isolate):
    """采用 = 增量并入（[n_adopted:] 深拷贝含 user_attrs）+ objective 盖章 + 目标变重开。"""
    import optuna
    from mflowy.model import session_remote

    def _mk(n, score):
        db = tmp_path / f"r{n}.db"
        st = optuna.create_study(study_name="study", storage=f"sqlite:///{db}", direction="minimize")
        for i in range(n):
            t = st.ask()
            t.set_user_attr("scores", [float(score + i)])
            st.tell(t, float(score + i))
        return db, st

    r1, st1 = _mk(2, 1.0)
    session_db = tmp_path / "s.db"
    adopted, n = session_remote.adopt_study_trials(
        session_db,
        study_name="k",
        objective_id={"seed": 1, "scoring": "mae"},
        direction="minimize",
        seed=1,
        remote_db=r1,
        n_adopted=0,
        remote_study_name="study",
    )
    assert n == 2 and len(adopted.trials) == 2
    assert adopted.user_attrs["objective"] == {"seed": 1, "scoring": "mae"}
    assert adopted.best_trial.user_attrs.get("scores")  # user_attrs 深拷贝保持

    # 增量：扩到 4 trial 的远程库，n_adopted=2 → 只并 2 条新
    r2, _ = _mk(4, 0.5)
    adopted2, n2 = session_remote.adopt_study_trials(
        session_db,
        study_name="k",
        objective_id={"seed": 1, "scoring": "mae"},
        direction="minimize",
        seed=1,
        remote_db=r2,
        n_adopted=2,
        remote_study_name="study",
    )
    assert n2 == 4 and len(adopted2.trials) == 4

    # 目标变（scoring 换）→ 重开（trial 清零）
    _, n3 = session_remote.adopt_study_trials(
        session_db,
        study_name="k",
        objective_id={"seed": 1, "scoring": "rmse"},
        direction="minimize",
        seed=1,
        remote_db=r2,
        n_adopted=0,
        remote_study_name="study",
    )
    assert n3 == 4
    reopened = optuna.load_study(study_name="k", storage=f"sqlite:///{session_db}")
    assert len(reopened.trials) == 4 and reopened.user_attrs["objective"] == {"seed": 1, "scoring": "rmse"}


# ── 终训 remote 分支（#218 门拆除——#156「终训恒 local」翻转，#225 裁定 5）──────


def _build_training_workspace(ws: Path) -> None:
    """终训形态工作区：全 pin RF（training 行为）——holdout 指标/副产物/final 模型
    同名工件（mlflow metrics 展平面 = holdout 解析面）。"""
    import mlflow
    from mflowy.data.types import TASKTYPE
    from mflowy.model.cv.split import split as split_fn
    from mflowy.model.model._split import materialize_split
    from mflowy.model.model.random_forest import RF
    from mflowy.model.x_y import XySpec

    ws.mkdir(parents=True, exist_ok=True)
    prev = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{ws / 'mlflow.db'}")
    try:
        mlflow.create_experiment("remote-final", artifact_location=str(ws / ".mlruns"))
        exp = mlflow.get_experiment_by_name("remote-final")
        assert exp is not None  # 刚创建（Optional 收窄）
        rng = np.random.default_rng(4)
        x = rng.normal(size=60)
        df = pd.DataFrame({"x": x, "target": 2.0 * x + 0.1 * rng.normal(size=60)})
        X, y = df[["x"]], df[["target"]]
        assert isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame)  # 列清单索引恒 DataFrame（联合噪声）
        xy = XySpec(X, y, TASKTYPE.REGRESSION)
        ts = materialize_split(split_fn(x_y=xy, cv=3), xy)
        with mlflow.start_run(experiment_id=exp.experiment_id, run_name="终局训练-RF"):
            RF(
                ts,
                n_estimators=10,
                max_depth=3,
                min_samples_split=2,
                min_samples_leaf=2,
                max_features="sqrt",
                ccp_alpha=0.0,
            )
            mlflow.set_tag("mflowy.node", "model__RF")
    finally:
        mlflow.set_tracking_uri(prev)


def test_final_train_test_submitted_branch(tmp_path, monkeypatch, isolate, reg_csv):
    """终训 remote：submitted → 拉取解包 → holdout 指标解析 + 副产物/模型镜像本地
    run → combo 冻结（账本两分支同形；loader 指镜像 run 的 runs:/ URI）。"""
    observation_json, _ = _start(reg_csv)
    ModelingSession().apply(observation_json, "select_model", {"family": "RF"})
    ModelingSession().apply(observation_json, "run_hpo", {"n_trials": 1})  # local 先搜（先搜后店）

    fake = FakePlatform(zips={})
    _install_remote_provider(monkeypatch, fake)
    ws = tmp_path / "remote-final"
    _build_training_workspace(ws)
    fake.zips["job-1"] = ws

    import mflowy.utils.constants as consts

    monkeypatch.setattr(consts, "REMOTE_TASK_DIR", str(ws))

    r = ModelingSession().apply(observation_json, "final_train_test", {"platform_experiment_id": "exp-1"})
    st = _state(observation_json)
    entry = st["models"][_K_RF]
    assert entry["frozen"] and entry["u"] < 0  # MAE 误差类取负（terminal_u 单源）
    assert entry["metrics"]["target"]["mae"] > 0 and entry["run_id"]
    assert entry["byproducts"] == {"X_test": "X_test.parquet", "y_test": "y_test.parquet", "y_pred": "y_pred.parquet"}
    assert r.biz["board"][_K_RF]["frozen"] is True

    # 镜像 run：本地库可寻址（副产物 + final 模型 + lineage tags）
    import mlflow
    from mflowy.utils.mlflow import run_artifacts

    arts = run_artifacts(entry["run_id"])
    assert {"X_test.parquet", "y_test.parquet", "y_pred.parquet", "final"} <= set(arts)
    tags = mlflow.get_run(entry["run_id"]).data.tags
    assert tags["mflowy.action"] == "final_train_test" and tags["mflowy.family"] == "RF"

    # loader：镜像 run 的 runs:/ URI（惰性 pipe 可载）+ 盘上世代文件
    from mflowy.model.session import load_loader

    loader = load_loader(Path(observation_json).parent.name, _K_RF)
    assert loader.holdout["u"] == entry["u"]
    assert loader.x_names == ["x"]

    # plan 审计件：uploads 寻址 + cv.split 复现形态
    plan = (Path(observation_json).parent / "plans" / "final_RF__passthrough__passthrough_v1.yaml").read_text()
    assert f"/user-data/uploads/{Path(reg_csv).name}" in plan and "cv__split" in plan


# ── 键控合并（#225 裁定 3——并行分支：entry[key]/steps/决策日志交换面）────────


def test_commit_keyed_merge_preserves_parallel_branch_entries(tmp_path, isolate, reg_csv):
    """board 键控合并：分支 A（配置编辑，body 持开）commit 时锁内重读——分支 B
    完整落账的 models 条目不被 A 的整写覆盖（旧整写形态即此缺陷）。"""
    from mflowy.model import session as ms

    observation_json, _ = _start(reg_csv)  # 候选 RF baseline
    ModelingSession().apply(observation_json, "select_model", {"family": "RF"})
    ModelingSession().apply(observation_json, "run_hpo", {"n_trials": 1})

    ss_a = ms._open(observation_json)
    with ss_a.apply_ctx("select_model") as (o_a, log_a):
        board_a = ms._board_load(ss_a)
        m_a = ms._mview(ss_a, o_a, board_a)
        # 分支 B：完整提交（report_findings on RF——账本条目落盘）
        ModelingSession().apply(
            observation_json,
            "report_findings",
            {"key": _K_RF, "findings": [{"kind": "error_concentrated", "severity": 0.5, "message": "并行分支落诊"}]},
        )
        # 分支 A 后提交（锁内重读合并面）
        ms._select_model(ss_a, o_a, m_a, board_a, log_a, {"family": "RF"})

    st = _state(observation_json)
    entry = st["models"][_K_RF]
    assert entry["findings"], f"分支 B 的落诊条目被 A 的 commit 覆盖：{entry.get('findings')}"
    assert entry["findings"][0]["source"] == "llm" and entry["findings"][0]["message"] == "并行分支落诊"
    assert st["cursor"]["m"] == "RF"  # A 的光标提交在（last-write-wins）
