"""Host-side DeLM HTTP board lifecycle and original-file capture per CooperBench pair."""

from __future__ import annotations

import io
import json
import re
import secrets
import socket
import subprocess
import tarfile
import threading
import uuid
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from board.protected_board import ProtectedBoard
from board.shared_context import BoardError, OriginalReadError
from cooperbench.utils import get_image_name

BOARD_WRITE_ACTIONS = frozenset({"status", "publish", "claim", "reserve", "done", "note", "apply"})
SUBJECT_MAX_CHARS = 80
TITLE_PATTERNS = (
    re.compile(r"^\*{0,2}Title\*{0,2}\s*:\s*(.+)$", re.IGNORECASE),
    re.compile(r"^\*{0,2}Feature\s*:\s*(.+?)\*{0,2}$", re.IGNORECASE),
    re.compile(r"^#+\s*(.+)$"),
)

_REGISTRY_LOCK = threading.Lock()
_RUNTIME_REGISTRY: dict[Path, PairBoardRuntime] = {}


def get_pair_runtime(log_dir: Path | str) -> PairBoardRuntime:
    """Return the active PairBoardRuntime registered for ``log_dir``."""
    resolved = Path(log_dir).resolve()
    with _REGISTRY_LOCK:
        runtime = _RUNTIME_REGISTRY.get(resolved)
    if runtime is None:
        raise RuntimeError(f"No active PairBoardRuntime registered for {resolved}")
    return runtime


class _BoardHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class OriginalFiles:
    """Lazy, tamper-proof reader for unmodified files in a task Docker image."""

    def __init__(self, image: str, workdir: str = "/workspace/repo") -> None:
        self.name = f"delm-original-{uuid.uuid4().hex[:12]}"
        self.workdir = workdir.rstrip("/")
        self._cache: dict[str, bytes | None] = {}
        self._cache_lock = threading.Lock()
        subprocess.run(
            ["docker", "create", "--name", self.name, "--label", "delm.workspace.research=true", image, "true"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def fetch(self, path: str) -> bytes | None:
        """Return original file bytes, None when absent in base image, or raise OriginalReadError."""
        with self._cache_lock:
            if path not in self._cache:
                self._cache[path] = self._fetch(path)
            return self._cache[path]

    def remove(self) -> None:
        """Force-remove the stopped base container."""
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, timeout=30)

    def _fetch(self, path: str) -> bytes | None:
        target = path if path.startswith("/") else f"{self.workdir}/{path}"
        try:
            result = subprocess.run(["docker", "cp", f"{self.name}:{target}", "-"], capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise OriginalReadError(f"Could not read {target}: {error}") from error
        if result.returncode:
            message = result.stderr.decode("utf-8", "replace").strip()
            if "Could not find the file" in message or "No such container:path" in message:
                return None
            raise OriginalReadError(f"Could not read {target} (docker cp exit {result.returncode}): {message}")
        try:
            with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
                members = archive.getmembers()
                if len(members) != 1 or not members[0].isfile() or members[0].name != target.rsplit("/", 1)[-1]:
                    raise OriginalReadError(f"Original {target} was not returned as one plain file")
                extracted = archive.extractfile(members[0])
                if extracted is None:
                    raise OriginalReadError(f"Original {target} could not be extracted")
                with extracted:
                    content = extracted.read()
                if len(content) != members[0].size:
                    raise OriginalReadError(f"Original {target} was not read completely")
                return content
        except (tarfile.TarError, OSError, EOFError) as error:
            raise OriginalReadError(f"Could not read original archive for {target}: {error}") from error


class PairBoardRuntime:
    """Context manager owning one pair's DeLM SQLite board, HTTP server, and OriginalFiles."""

    def __init__(
        self,
        *,
        pair_log_dir: Path | str,
        repo_name: str,
        task_id: int,
        features: list[int] | tuple[int, int],
        dataset_dir: Path | str,
        actors: tuple[str, ...] = ("agent1", "agent2"),
    ) -> None:
        self.pair_log_dir = Path(pair_log_dir).resolve()
        self.repo_name = repo_name
        self.task_id = int(task_id)
        self.sorted_features = sorted(int(f) for f in features)
        if len(self.sorted_features) != 2:
            raise ValueError(f"Expected 2 features for a CooperBench pair, got {features}")
        self.dataset_dir = Path(dataset_dir).resolve()
        self.actors = tuple(actors)
        if len(self.actors) != 2:
            raise ValueError(f"Expected 2 actors, got {actors}")

        self.delm_dir = self.pair_log_dir / "delm"
        self.db_path = self.delm_dir / "context.sqlite"
        self.records_dirs = {actor: self.delm_dir / "records" / actor for actor in self.actors}
        self.tokens = {secrets.token_hex(16): actor for actor in self.actors}
        self._actor_to_token = {actor: token for token, actor in self.tokens.items()}

        self.original_files: OriginalFiles | None = None
        self.board: ProtectedBoard | None = None
        self.server: _BoardHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self.port: int = 0
        self.context_url: str = ""
        self.item_ids: dict[str, str] = {}
        self._preseed_max_sequence: int = 0

    def __enter__(self) -> PairBoardRuntime:
        try:
            self.delm_dir.mkdir(parents=True, exist_ok=True)
            for rec_dir in self.records_dirs.values():
                rec_dir.mkdir(parents=True, exist_ok=True)
                rec_dir.chmod(0o777)
                err_file = rec_dir / "context-command-errors.jsonl"
                err_file.touch(exist_ok=True)
                err_file.chmod(0o666)

            image = get_image_name(self.repo_name, self.task_id)
            self.original_files = OriginalFiles(image=image, workdir="/workspace/repo")
            self.board = ProtectedBoard(
                self.db_path,
                actors=self.actors,
                original=self.original_files.fetch,
            )
            self._prepopulate_board()

            handler_cls = _make_board_handler(self.board, self.tokens)
            self.server = _BoardHTTPServer(("0.0.0.0", 0), handler_cls)
            self.port = int(self.server.server_address[1])
            self.context_url = f"http://host.docker.internal:{self.port}"
            self._server_thread = threading.Thread(
                target=self.server.serve_forever,
                name=f"delm-board-{self.repo_name}-{self.task_id}",
                daemon=True,
            )
            self._server_thread.start()

            with _REGISTRY_LOCK:
                _RUNTIME_REGISTRY[self.pair_log_dir] = self
            return self
        except Exception:
            self._cleanup_resources()
            raise

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            self.export_events()
        finally:
            self._cleanup_resources()

    def token_for(self, actor: str) -> str:
        """Return the secret HTTP token assigned to ``actor``."""
        if actor not in self._actor_to_token:
            raise KeyError(f"Unknown actor: {actor}")
        return self._actor_to_token[actor]

    def records_dir(self, actor: str) -> Path:
        """Return the writable host directory bind-mounted to ``/records`` for ``actor``."""
        if actor not in self.records_dirs:
            raise KeyError(f"Unknown actor: {actor}")
        return self.records_dirs[actor]

    def mark_session(self, actor: str, state: str) -> None:
        """Record ``running`` or ``ended`` session lifecycle on the board."""
        if self.board is not None:
            self.board.session_state(actor, state)

    def export_events(self) -> dict[str, Any]:
        """Export board events and file metadata (without raw file bodies) to ``delm_board_events.json``."""
        if self.board is None:
            return {}
        events = self.board.export()
        raw_bodies = self.board.export_bodies()
        sanitized_bodies = [
            {
                "id": body["id"],
                "time_ns": body["time_ns"],
                "actor": body["actor"],
                "files": [
                    {
                        k: v
                        for k, v in f.items()
                        if k not in ("content", "original")
                    }
                    for f in body.get("files", [])
                ],
            }
            for body in raw_bodies
        ]
        summary = summarize_board_events(
            events=events,
            bodies=sanitized_bodies,
            actors=self.actors,
            preseed_max_sequence=self._preseed_max_sequence,
        )
        payload = {
            "repo": self.repo_name,
            "task_id": self.task_id,
            "features": self.sorted_features,
            "item_ids": self.item_ids,
            "preseed_max_sequence": self._preseed_max_sequence,
            "summary": summary,
            "events": events,
            "bodies": sanitized_bodies,
        }
        self.pair_log_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.pair_log_dir / "delm_board_events.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return payload

    def _prepopulate_board(self) -> None:
        assert self.board is not None
        task_dir = self.dataset_dir / self.repo_name / f"task{self.task_id}"
        f1, f2 = self.sorted_features
        spec1 = (task_dir / f"feature{f1}" / "feature.md").read_text(encoding="utf-8")
        spec2 = (task_dir / f"feature{f2}" / "feature.md").read_text(encoding="utf-8")
        subject1 = _format_item_subject(f1, spec1)
        subject2 = _format_item_subject(f2, spec2)
        detail1 = spec1.strip()
        detail2 = spec2.strip()

        with self.board.connect() as db:
            item1_id = self.board.write_item(db, "system", subject1, detail1)
            item2_id = self.board.write_item(db, "system", subject2, detail2)
            self.board.append(db, self.actors[0], "claim", {"item": item1_id})
            self.board.append(db, self.actors[0], "status", {"text": subject1, "auto": True})
            self.board.append(db, self.actors[1], "claim", {"item": item2_id})
            self.board.append(db, self.actors[1], "status", {"text": subject2, "auto": True})
            if self.board.last_read(db, self.actors[0]) is not None or self.board.last_read(db, self.actors[1]) is not None:
                raise RuntimeError("Pre-population unexpectedly appended a read event")
            self._preseed_max_sequence = int(self.board.last_sequence(db))
        self.item_ids = {self.actors[0]: item1_id, self.actors[1]: item2_id}

    def _cleanup_resources(self) -> None:
        with _REGISTRY_LOCK:
            _RUNTIME_REGISTRY.pop(self.pair_log_dir, None)
        if self.board is not None:
            try:
                self.board.stop()
            except Exception:
                pass
        if self.server is not None:
            try:
                self.server.shutdown()
            except Exception:
                pass
            try:
                self.server.server_close()
            except Exception:
                pass
        if self._server_thread is not None and self._server_thread.is_alive():
            self._server_thread.join(timeout=5)
        if self.original_files is not None:
            try:
                self.original_files.remove()
            except Exception:
                pass


def summarize_board_events(
    *,
    events: list[dict[str, Any]],
    bodies: list[dict[str, Any]],
    actors: tuple[str, ...] = ("agent1", "agent2"),
    preseed_max_sequence: int = 0,
) -> dict[str, Any]:
    """Compute per-kind, per-actor, and write counts from exported board events."""
    by_kind: Counter[str] = Counter()
    by_actor_kind: dict[str, Counter[str]] = {actor: Counter() for actor in actors}
    command_counts: dict[str, Counter[str]] = {actor: Counter() for actor in actors}

    for ev in events:
        seq = int(ev.get("sequence", 0))
        actor = ev.get("actor", "")
        kind = ev.get("kind", "")
        payload = ev.get("payload") or {}
        by_kind[kind] += 1
        if seq <= preseed_max_sequence:
            continue
        if actor in by_actor_kind:
            by_actor_kind[actor][kind] += 1
            if kind == "read":
                via = payload.get("via", "read")
                if via == "apply_result":
                    command_counts[actor]["apply"] += 1
                elif via != "wait":
                    command_counts[actor][via] += 1
            elif kind == "wait":
                command_counts[actor]["wait"] += 1
            elif kind == "apply" and not payload.get("applied", True):
                command_counts[actor]["apply_conflict"] += 1

    per_actor_writes = {
        actor: sum(command_counts[actor][act] for act in BOARD_WRITE_ACTIONS)
        for actor in actors
    }
    return {
        "total_events": len(events),
        "total_bodies": len(bodies),
        "board_writes_total": sum(per_actor_writes.values()),
        "board_writes_by_actor": per_actor_writes,
        "by_kind": dict(by_kind),
        "by_actor_kind": {actor: dict(cnt) for actor, cnt in by_actor_kind.items()},
        "commands_by_actor": {actor: dict(cnt) for actor, cnt in command_counts.items()},
    }


def _extract_title(spec: str) -> str:
    lines = [line.strip() for line in spec.splitlines() if line.strip()]
    for line in lines[:8]:
        for pattern in TITLE_PATTERNS:
            match = pattern.match(line)
            if match:
                return _clean_title(match.group(1))
    return _clean_title(lines[0]) if lines else "Feature implementation"


def _clean_title(raw: str) -> str:
    cleaned = raw.replace("**", "").replace("`", "").strip()
    return " ".join(cleaned.split()) or "Feature implementation"


def _format_item_subject(feature_id: int, spec: str) -> str:
    prefix = f"Feature {feature_id}: "
    title = _extract_title(spec)
    full = prefix + title
    if len(full) <= SUBJECT_MAX_CHARS:
        return full
    return full[: SUBJECT_MAX_CHARS - 3].rstrip() + "..."


def _make_board_handler(context: ProtectedBoard, actors: dict[str, str]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            del format, args

        def do_GET(self) -> None:
            self._reply(200, {"ready": True} if self.path == "/health" else {"error": "not found"})

        def do_POST(self) -> None:
            actor = actors.get(self.path.removeprefix("/"))
            if actor is None:
                self._reply(403, {"error": "unknown identity"})
                return
            try:
                content_len = int(self.headers.get("Content-Length", "0"))
                p = json.loads(self.rfile.read(content_len))
                action = p["action"]
                if action == "read":
                    answer = context.read(
                        actor,
                        full=p.get("all", False),
                        wait_seconds=p.get("wait") or 0,
                        target=p.get("for") or None,
                    )
                elif action == "status":
                    answer = context.status(actor, p["text"])
                elif action == "claim":
                    answer = context.claim(actor, p["item"])
                elif action == "reserve":
                    answer = context.reserve(actor, p["subject"], p["detail"])
                elif action == "done":
                    answer = context.done(
                        actor,
                        p["item"],
                        p["summary"],
                        p.get("files"),
                        request_id=p.get("request_id"),
                    )
                elif action == "note":
                    answer = context.note(
                        actor,
                        p["type"],
                        p["text"],
                        p["scope"],
                        p["basis"],
                        p.get("files"),
                        request_id=p.get("request_id"),
                    )
                elif action == "attached":
                    answer = context.attached(actor, p["id"])
                elif action == "expand":
                    answer = context.expand(actor, p["id"], p.get("paths"))
                    answer["view"] = context.view_after(actor, "expand")
                elif action == "apply":
                    answer = context.apply(actor, p["id"], p["local"], p.get("paths"))
                    if not answer["applied"]:
                        answer["view"] = context.view_after(actor, "apply")
                elif action == "apply-error":
                    context.apply_error(actor, p["id"], p["error"])
                    answer = {"saved": True}
                elif action == "apply-result":
                    answer = context.apply_result(actor, p["id"], p["written"], p["success"])
                elif action == "publish":
                    answer = context.publish(actor, p["items"])
                else:
                    raise BoardError("unknown action")
                self._reply(200, answer)
            except (BoardError, KeyError, ValueError, socket.timeout) as error:
                self._reply(400, {"error": str(error)})

        def _reply(self, code: int, body: dict[str, Any]) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler
