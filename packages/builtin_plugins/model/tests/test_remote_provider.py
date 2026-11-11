"""RemoteJobProvider 提交面契约（#156/#157/#158 决议，切片④）。

假 client 经构造缝注入（真实 client 详测住 test_platform_client.py）；preset 名
经 monkeypatch env；async 方法经 ``asyncio.run`` 驱动（仓内既有约定，无 asyncio 插件）。
断言面：提交 argv 形状、校验 fail-loud 家族、JobOutcome 形态。
"""

import asyncio
import json
from types import SimpleNamespace

import mflowy.model.job_provider as jp
import pytest
from mflowy.model.job_provider.protocol import JobOutcome
from mflowy.model.job_provider.remote import RemoteJobProvider

STEPS = "- x_y:\n    name: 拆分\n    params:\n      target: target\n"


class FakeClient:
    def __init__(self, *, outputs=(), presets=("cpu-standard",)):
        self.outputs = list(outputs)
        self.presets = list(presets)
        self.created: list[tuple] = []

    def list_presets(self):
        return [SimpleNamespace(id=f"preset-{n}", name=n) for n in self.presets]

    def list_files(self, experiment_id, mount_type="", prefix=""):
        from mflowy.model.job_provider.platform_client import FileEntry

        return [FileEntry(mount_type="outputs", filename=f, size=1, etag="e", last_modified=1) for f in self.outputs]

    def create_job(self, cmd, args, experiment_id, preset_id):
        self.created.append((cmd, args, experiment_id, preset_id))
        return SimpleNamespace(id="job-new-1", status="queued", created_at=1)


@pytest.fixture(autouse=True)
def _provider_reset():
    jp.set_job_provider(None)
    yield
    jp.set_job_provider(None)


@pytest.fixture
def preset_env(monkeypatch):
    monkeypatch.setenv("MFLOWY_PLATFORM_JOB_PRESET", "cpu-standard")


# ── 提交面形状（#158 exec 风格）──────────────────────────────────────────


def test_submit_face_and_outcome(preset_env):
    fake = FakeClient()
    provider = RemoteJobProvider(client=fake)
    outcome = asyncio.run(
        provider.hyperparameters_search(modeling_steps=STEPS, name="n", desc="d", platform_experiment_id="exp-1")
    )
    cmd, args, exp, preset_id = fake.created[0]
    assert cmd == "mflowy-job"
    assert args[:3] == ["wrap", "--tool", "hyperparameters_search"]
    assert args[3] == "--kwargs-json"
    payload = json.loads(args[4])
    assert payload["modeling_steps"] == STEPS and payload["name"] == "n"
    assert "--inherit-from" not in args  # 继承非空才出现
    assert exp == "exp-1" and preset_id == "preset-cpu-standard"
    assert outcome == JobOutcome.submitted(job_id="job-new-1", platform_experiment_id="exp-1")
    assert outcome.dag_result is None  # remote 永不阻塞等待


def test_submit_with_inherit_appends_flag(preset_env):
    fake = FakeClient(outputs=["job-old.zip"])
    provider = RemoteJobProvider(client=fake)
    asyncio.run(
        provider.modeling(
            modeling_steps=STEPS,
            name="n",
            desc="d",
            experiment_id="mlflow-exp-9",
            platform_experiment_id="exp-1",
            inherit_platform_job_id="job-old",
        )
    )
    args = fake.created[0][1]
    assert args[-2:] == ["--inherit-from", "job-old"]
    assert json.loads(args[4])["experiment_id"] == "mlflow-exp-9"


# ── 校验 fail-loud 家族（#156 §2）─────────────────────────────────────────


def test_missing_platform_experiment_id_points_to_ensure(preset_env):
    provider = RemoteJobProvider(client=FakeClient())
    with pytest.raises(ValueError, match="ensure_platform_experiment"):
        asyncio.run(provider.hyperparameters_search(modeling_steps=STEPS, name="n", desc="d"))


def test_preset_name_resolution_failures(monkeypatch):
    monkeypatch.delenv("MFLOWY_PLATFORM_JOB_PRESET", raising=False)
    provider = RemoteJobProvider(client=FakeClient())
    with pytest.raises(ValueError, match="MFLOWY_PLATFORM_JOB_PRESET 未设"):
        asyncio.run(
            provider.hyperparameters_search(modeling_steps=STEPS, name="n", desc="d", platform_experiment_id="e")
        )

    monkeypatch.setenv("MFLOWY_PLATFORM_JOB_PRESET", "no-such")
    with pytest.raises(ValueError, match=r"不命中.*cpu-standard"):
        asyncio.run(
            provider.hyperparameters_search(modeling_steps=STEPS, name="n", desc="d", platform_experiment_id="e")
        )


def test_preset_resolved_fresh_per_submission(preset_env):
    """append-only 版本链——每次提交 fresh list_presets()（不缓存钉旧版）。"""
    fake = FakeClient()
    provider = RemoteJobProvider(client=fake)
    asyncio.run(provider.hyperparameters_search(modeling_steps=STEPS, name="n", desc="d", platform_experiment_id="e"))
    fake.presets.append("gpu-standard")  # 平台侧新增版本链
    asyncio.run(provider.hyperparameters_search(modeling_steps=STEPS, name="n", desc="d", platform_experiment_id="e"))
    assert fake.created[0][3] == "preset-cpu-standard" and fake.created[1][3] == "preset-cpu-standard"


def test_inherit_zip_missing_lists_candidates(preset_env):
    fake = FakeClient(outputs=["job-a.zip", "job-b.zip"])
    provider = RemoteJobProvider(client=fake)
    with pytest.raises(ValueError, match=r"job-gone.*job-a\.zip.*job-b\.zip"):
        asyncio.run(
            provider.modeling(
                modeling_steps=STEPS,
                name="n",
                desc="d",
                experiment_id="x",
                platform_experiment_id="e",
                inherit_platform_job_id="job-gone",
            )
        )


def test_prior_state_requires_inherit_static_gate(preset_env):
    provider = RemoteJobProvider(client=FakeClient(outputs=["job-old.zip"]))
    with pytest.raises(ValueError, match="inherit_platform_job_id"):
        asyncio.run(
            provider.explanation(modeling_steps=STEPS, model="run-1", name="n", desc="d", platform_experiment_id="e")
        )
    with pytest.raises(ValueError, match="inherit_platform_job_id"):
        asyncio.run(
            provider.modeling(
                modeling_steps=STEPS, name="n", desc="d", experiment_id="mlflow-1", platform_experiment_id="e"
            )
        )
    # hpo + study_storage_path 不强制（#156：不继承 = 新 study）
    asyncio.run(
        provider.hyperparameters_search(
            modeling_steps=STEPS, name="n", desc="d", study_storage_path="/tmp/s.db", platform_experiment_id="e"
        )
    )


def test_kwargs_json_over_128kib_rejected(preset_env):
    provider = RemoteJobProvider(client=FakeClient())
    with pytest.raises(ValueError, match="128KiB"):
        asyncio.run(
            provider.modeling(modeling_steps="x" * (128 * 1024), name="n", desc="d", platform_experiment_id="e")
        )


# ── 工厂解析（MFLOWY_JOB_PROVIDER=remote）─────────────────────────────────


def test_factory_resolves_remote(monkeypatch):
    monkeypatch.setenv("MFLOWY_JOB_PROVIDER", "remote")
    assert type(jp.get_job_provider()).__name__ == "RemoteJobProvider"
