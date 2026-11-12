"""CooperBench agent adapter running Mini-SWE-Agent v2 with a DeLM shared-context board."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from cooperbench.agents import AgentResult
from cooperbench.agents._azure import azure_litellm_model, resolve_azure_config
from cooperbench.agents._coop.runtime import normalize_patch
from cooperbench.agents.mini_swe_agent_v2.agents.default import GIT_REMOTE, DefaultAgent
from cooperbench.agents.mini_swe_agent_v2.config import get_config_path
from cooperbench.agents.mini_swe_agent_v2.connectors import GitConnector
from cooperbench.agents.mini_swe_agent_v2.environments.docker import DockerEnvironment
from cooperbench.agents.mini_swe_agent_v2.models.litellm_model import LitellmModel
from cooperbench.agents.mini_swe_agent_v2.utils.serialize import recursive_merge
from cooperbench.agents.registry import register
from delm_cb.board_runtime import get_pair_runtime
from delm_cb.prompts import DELM_INSTANCE_TEMPLATE, DELM_SYSTEM_TEMPLATE

logger = logging.getLogger(__name__)
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
VENDOR_BOARD_DIR = PACKAGE_ROOT / "vendor" / "delm" / "board"


class DelmDefaultAgent(DefaultAgent):
    """DefaultAgent subclass that preserves unsubmitted/unpushed PR nudges without Redis comm."""

    def __init__(self, *args: Any, delm_agent_id: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._delm_agent_id = delm_agent_id or self.agent_id

    def _opened_pr(self) -> bool:
        result = self.env.execute(
            {"command": f"git ls-remote --tags {GIT_REMOTE} refs/tags/pr/{self._delm_agent_id}"}
        )
        return bool((result.get("output") or "").strip())

    def _nudge_unsubmitted(self) -> bool:
        if getattr(self, "_submit_nudges", 0) >= self.MAX_SUBMIT_NUDGES or self._opened_pr():
            return False
        agent = self._delm_agent_id
        dirty = (self.env.execute({"command": "git status --porcelain"}).get("output") or "").strip()
        ahead = (
            self.env.execute({"command": f"git rev-list --count {GIT_REMOTE}/{agent}..HEAD 2>/dev/null"})
            .get("output", "")
            .strip()
        )
        unpushed = ahead not in ("", "0")
        if not (dirty or unpushed):
            return False
        self._submit_nudges = getattr(self, "_submit_nudges", 0) + 1
        held = "uncommitted changes" if dirty else "commits that are not pushed"
        self.log(f"SUBMIT NUDGE {self._submit_nudges} for {agent}: {held}, no PR")
        self.add_messages(
            self.model.format_message(
                role="user",
                content=(
                    f"You have not submitted anything. Your work is still local ({held}) and "
                    f"no pull request exists, so it would be graded as an empty submission.\n\n"
                    f"    git add <the files you want to submit>\n"
                    f'    git commit -m "..."\n'
                    f"    git push {GIT_REMOTE} HEAD:{agent}\n"
                    f'    gh pr create --title "..." --body "..."\n\n'
                    f"Then echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT again."
                ),
            )
        )
        return True


@register("delm_msa_v2")
class DelmMiniSweAgentV2Runner:
    """CooperBench AgentRunner wiring Mini-SWE-Agent v2 containers to a pair's DeLM board."""

    def run(
        self,
        task: str,
        image: str,
        *,
        agent_id: str = "agent",
        model_name: str = "vertex_ai/gemini-3.8-flash",
        agents: list[str] | None = None,
        comm_url: str | None = None,
        git_server_url: str | None = None,
        git_enabled: bool = False,
        messaging_enabled: bool = False,
        config: dict[str, Any] | None = None,
        agent_config: str | None = None,
        log_dir: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        del comm_url, messaging_enabled, kwargs
        if not log_dir:
            raise RuntimeError("DelmMiniSweAgentV2Runner requires log_dir matching a registered PairBoardRuntime")
        runtime = get_pair_runtime(log_dir)

        default_config = _load_merged_config(
            is_coop=bool(agents) and len(agents) > 1,
            agent_config=agent_config,
            config=config,
        )
        agent_cfg = dict(default_config.get("agent", {}))
        agent_cfg["system_template"] = DELM_SYSTEM_TEMPLATE
        agent_cfg["instance_template"] = DELM_INSTANCE_TEMPLATE
        model_cfg = dict(default_config.get("model", {}))
        env_cfg = dict(default_config.get("environment", {}))

        backend = default_config.get("backend", "docker")
        if backend != "docker":
            raise RuntimeError(f"delm_msa_v2 supports docker backend only, got {backend!r}")

        records_dir = runtime.records_dir(agent_id)
        (records_dir / "prompt.txt").write_text(task, encoding="utf-8")

        container_env = dict(env_cfg.get("env") or {})
        container_env.update(
            {
                "CONTEXT_URL": runtime.context_url,
                "CONTEXT_TOKEN": runtime.token_for(agent_id),
                "CONTEXT_PYTHON": "python3",
                "CONTEXT_LIBRARY_PATH": "",
                "TASK_WORKSPACE": "/workspace/repo",
            }
        )
        run_args = [
            "--rm",
            "--add-host=host.docker.internal:host-gateway",
            "-v",
            f"{VENDOR_BOARD_DIR}:/tools:ro",
            "-v",
            f"{records_dir}:/records:rw",
        ]
        env_kwargs: dict[str, Any] = {
            "image": image,
            "cwd": "/workspace/repo",
            "timeout": 10800,
            "env": container_env,
            "run_args": run_args,
        }
        if env_cfg.get("container_timeout"):
            env_kwargs["container_timeout"] = str(env_cfg["container_timeout"])
        if config and config.get("git_network"):
            env_kwargs["network"] = config["git_network"]

        env = DockerEnvironment(**env_kwargs)
        error_msg: str | None = None
        patch = ""
        status = "Error"
        agent: DelmDefaultAgent | None = None

        try:
            azure = resolve_azure_config()
            if azure:
                model_name = azure_litellm_model(model_name)
                model_kwargs = dict(model_cfg.get("model_kwargs") or {})
                model_kwargs.update({"api_base": azure["endpoint"], "api_key": azure["api_key"]})
                model_cfg["model_kwargs"] = model_kwargs
            model = LitellmModel(model_name=model_name, **model_cfg)

            git_connector = GitConnector(
                agent_id=agent_id,
                agents=agents or [agent_id],
                server_url=git_server_url if (git_enabled and git_server_url) else "",
            )
            git_connector.setup(env)

            f1, f2 = runtime.sorted_features
            agent = DelmDefaultAgent(
                model=model,
                env=env,
                comm=None,
                agent_id=agent_id,
                delm_agent_id=agent_id,
                **agent_cfg,
            )
            agent.extra_template_vars.update(
                {
                    "agent_id": agent_id if (agents and len(agents) > 1) else None,
                    "agents": agents if agents else [],
                    "git_enabled": git_enabled,
                    "messaging_enabled": False,
                    "delm_item1_id": runtime.item_ids["agent1"],
                    "delm_item2_id": runtime.item_ids["agent2"],
                    "delm_f1": f1,
                    "delm_f2": f2,
                }
            )

            runtime.mark_session(agent_id, "running")
            try:
                result = agent.run(task=task)
                status = result.get("exit_status", "Submitted")
            except Exception as exc:
                status = "Error"
                error_msg = str(exc)
            finally:
                runtime.mark_session(agent_id, "ended")

            try:
                patch = normalize_patch(git_connector.submitted_patch(env))
            except Exception:
                patch = ""

            if agent._compaction_count > 0:
                traj_path = Path(log_dir) / f"{agent_id}_full_traj.json"
                agent.save(traj_path)
                logger.info(
                    "[%s] Full trajectory saved to %s (%d compaction(s))",
                    agent_id,
                    traj_path,
                    agent._compaction_count,
                )
        finally:
            env.cleanup()

        sanitized_messages = [
            {**msg, "content": ""} if msg.get("content") is None else msg
            for msg in (agent.messages if agent is not None else [])
        ]
        return AgentResult(
            status=status,
            patch=patch,
            cost=agent.cost if agent is not None else 0.0,
            steps=agent.n_calls if agent is not None else 0,
            messages=sanitized_messages,
            sent_messages=agent.sent_messages if agent is not None else [],
            error=error_msg,
        )


def _load_merged_config(*, is_coop: bool, agent_config: str | None, config: dict[str, Any] | None) -> dict[str, Any]:
    config_path = get_config_path("coop" if is_coop else "solo")
    with open(config_path, encoding="utf-8") as f:
        merged = yaml.safe_load(f) or {}
    if agent_config:
        with open(agent_config, encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}
        merged = recursive_merge(merged, overrides.get("config", overrides))
    if config is not None:
        merged = recursive_merge(merged, config)
    return merged
