import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.schemas.response import Message
from app.utils.common_utils import create_work_dir, get_work_dir
from app.utils.log_util import logger


class FileTaskStore:
    """Local task store used by the backend-only service.

    The module keeps the old import path (`redis_manager`) so the agent and
    workflow layers can stay largely unchanged while we remove Redis/WebSocket
    dependencies from the runtime architecture.
    """

    def __init__(self) -> None:
        self.project_dir = Path("project")
        self.project_dir.mkdir(parents=True, exist_ok=True)

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _ensure_work_dir(self, task_id: str) -> Path:
        work_dir = Path(create_work_dir(task_id))
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    def _work_dir_path(self, task_id: str) -> Path:
        return Path("project") / "work_dir" / task_id

    def _status_path(self, task_id: str) -> Path:
        return self._work_dir_path(task_id) / "status.json"

    def _messages_path(self, task_id: str) -> Path:
        return self._work_dir_path(task_id) / "messages.jsonl"

    def _request_path(self, task_id: str) -> Path:
        return self._work_dir_path(task_id) / "request.json"

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def task_exists(self, task_id: str) -> bool:
        try:
            return self._status_path(task_id).exists()
        except Exception:
            return False

    async def initialize_task(self, task_id: str, payload: dict[str, Any] | None = None) -> None:
        work_dir = self._ensure_work_dir(task_id)
        now = self._utc_now()
        status_payload = {
            "task_id": task_id,
            "status": "queued",
            "stage": "queued",
            "message": "Task created",
            "error": None,
            "created_at": now,
            "updated_at": now,
            "work_dir": str(work_dir),
            "artifacts": [],
        }
        self._write_json(self._status_path(task_id), status_payload)

        if payload is not None:
            request_payload = {
                "task_id": task_id,
                "created_at": now,
                "payload": payload,
            }
            self._write_json(self._request_path(task_id), request_payload)

        messages_path = self._messages_path(task_id)
        if not messages_path.exists():
            messages_path.touch()

    async def update_task(
        self,
        task_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        message: str | None = None,
        error: str | None = None,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        current = self._read_json(self._status_path(task_id))
        if not current:
            await self.initialize_task(task_id)
            current = self._read_json(self._status_path(task_id))

        if status is not None:
            current["status"] = status
        if stage is not None:
            current["stage"] = stage
        if message is not None:
            current["message"] = message
        if error is not None:
            current["error"] = error
        if artifacts is not None:
            current["artifacts"] = artifacts
        current["updated_at"] = self._utc_now()

        self._write_json(self._status_path(task_id), current)
        return current

    async def set(self, key: str, value: str) -> None:
        """Backward-compatible shim for the previous Redis `set` usage."""
        if key.startswith("task_id:"):
            await self.initialize_task(value)

    async def publish_message(self, task_id: str, message: Message) -> None:
        if not self.task_exists(task_id):
            await self.initialize_task(task_id)

        messages_path = self._messages_path(task_id)
        with messages_path.open("a", encoding="utf-8") as f:
            f.write(message.model_dump_json())
            f.write("\n")

        await self.update_task(task_id, message=message.content or "Task updated")
        logger.debug(f"Task message saved: task_id={task_id} msg_type={message.msg_type}")

    async def get_task_status(self, task_id: str) -> dict[str, Any]:
        status = self._read_json(self._status_path(task_id))
        if not status:
            raise FileNotFoundError(f"Task not found: {task_id}")
        return status

    async def get_task_request(self, task_id: str) -> dict[str, Any]:
        request_payload = self._read_json(self._request_path(task_id))
        if not request_payload:
            raise FileNotFoundError(f"Task request not found: {task_id}")
        return request_payload

    async def get_task_messages(self, task_id: str) -> list[dict[str, Any]]:
        messages_path = self._messages_path(task_id)
        if not messages_path.exists():
            return []

        messages: list[dict[str, Any]] = []
        with messages_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                messages.append(json.loads(line))
        return messages

    async def list_artifacts(self, task_id: str) -> list[dict[str, Any]]:
        work_dir = Path(get_work_dir(task_id))
        ignored_names = {"status.json", "messages.jsonl", "request.json"}

        artifacts: list[dict[str, Any]] = []
        for path in sorted(work_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.name in ignored_names:
                continue

            rel_path = path.relative_to(work_dir).as_posix()
            artifacts.append(
                {
                    "name": path.name,
                    "path": rel_path,
                    "size": path.stat().st_size,
                }
            )

        return artifacts


task_store = FileTaskStore()
# Backward compatible alias for existing imports.
redis_manager = task_store
