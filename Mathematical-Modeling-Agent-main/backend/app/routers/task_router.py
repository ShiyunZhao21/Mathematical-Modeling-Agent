from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from app.config.setting import settings
from app.core.workflow import MathModelWorkFlow
from app.schemas.enums import CompTemplate, FormatOutPut
from app.schemas.request import Problem
from app.services.redis_manager import task_store
from app.utils.common_utils import create_task_id, create_work_dir
from app.utils.log_util import logger


router = APIRouter(prefix="/tasks", tags=["tasks"])


def _artifact_url(task_id: str, rel_path: str) -> str:
    return f"{settings.SERVER_HOST}/static/{task_id}/{rel_path}"


async def _save_uploaded_files(work_dir: str, files: list[UploadFile] | None) -> list[str]:
    saved_files: list[str] = []
    if not files:
        return saved_files

    for file in files:
        if not file.filename:
            continue

        safe_name = Path(file.filename).name
        destination = Path(work_dir) / safe_name
        content = await file.read()
        destination.write_bytes(content)
        saved_files.append(safe_name)

    return saved_files


async def run_task_async(problem: Problem) -> None:
    try:
        await task_store.update_task(
            problem.task_id,
            status="running",
            stage="workflow",
            message="Task started",
        )
        await MathModelWorkFlow().execute(problem)

        artifacts = await task_store.list_artifacts(problem.task_id)
        await task_store.update_task(
            problem.task_id,
            status="completed",
            stage="completed",
            message="Task completed",
            artifacts=artifacts,
        )
    except Exception as exc:
        logger.exception(f"Task failed: task_id={problem.task_id} error={exc}")
        artifacts = []
        if task_store.task_exists(problem.task_id):
            artifacts = await task_store.list_artifacts(problem.task_id)

        await task_store.update_task(
            problem.task_id,
            status="failed",
            stage="failed",
            message="Task failed",
            error=str(exc),
            artifacts=artifacts,
        )


@router.post("/run")
async def run_task(
    background_tasks: BackgroundTasks,
    ques_all: str = Form(...),
    comp_template: CompTemplate = Form(default=CompTemplate.CHINA),
    format_output: FormatOutPut = Form(default=FormatOutPut.Markdown),
    files: list[UploadFile] | None = File(default=None),
):
    if format_output != FormatOutPut.Markdown:
        raise HTTPException(
            status_code=400,
            detail="The backend-only service currently supports Markdown -> PDF output only.",
        )

    task_id = create_task_id()
    work_dir = create_work_dir(task_id)
    saved_files = await _save_uploaded_files(work_dir, files)

    payload = {
        "ques_all": ques_all,
        "comp_template": comp_template.value,
        "format_output": format_output.value,
        "files": saved_files,
    }
    await task_store.initialize_task(task_id, payload=payload)

    problem = Problem(
        task_id=task_id,
        ques_all=ques_all,
        comp_template=comp_template,
        format_output=format_output,
    )
    background_tasks.add_task(run_task_async, problem)

    return {
        "task_id": task_id,
        "status": "queued",
        "stage": "queued",
        "saved_files": saved_files,
    }


@router.get("/{task_id}")
async def get_task(task_id: str):
    if not task_store.task_exists(task_id):
        raise HTTPException(status_code=404, detail="Task not found")

    status = await task_store.get_task_status(task_id)
    artifacts = await task_store.list_artifacts(task_id)
    for artifact in artifacts:
        artifact["url"] = _artifact_url(task_id, artifact["path"])

    status["artifacts"] = artifacts
    return status


@router.get("/{task_id}/messages")
async def get_task_messages(task_id: str):
    if not task_store.task_exists(task_id):
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task_id,
        "messages": await task_store.get_task_messages(task_id),
    }


@router.get("/{task_id}/artifacts")
async def get_task_artifacts(task_id: str):
    if not task_store.task_exists(task_id):
        raise HTTPException(status_code=404, detail="Task not found")

    artifacts = await task_store.list_artifacts(task_id)
    for artifact in artifacts:
        artifact["url"] = _artifact_url(task_id, artifact["path"])

    return {
        "task_id": task_id,
        "artifacts": artifacts,
    }


@router.get("/{task_id}/request")
async def get_task_request(task_id: str):
    if not task_store.task_exists(task_id):
        raise HTTPException(status_code=404, detail="Task not found")

    return await task_store.get_task_request(task_id)
