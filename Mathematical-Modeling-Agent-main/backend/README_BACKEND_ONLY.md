# MathModelAgent Backend Only

This backend-only version keeps the multi-agent workflow:

- `CoordinatorAgent`: restructure the modeling problem
- `ModelerAgent`: design modeling plans
- `CoderAgent`: execute Python code and generate figures
- `WriterAgent`: assemble the final paper

## Main API

### `POST /tasks/run`

Submit a modeling task with multipart form data:

- `ques_all`: full problem statement
- `comp_template`: `CHINA` (default)
- `format_output`: `Markdown` only
- `files`: optional dataset files

### `GET /tasks/{task_id}`

Get task status and artifact list.

### `GET /tasks/{task_id}/messages`

Get recorded workflow messages.

### `GET /tasks/{task_id}/artifacts`

Get artifact metadata and download URLs.

## Output Files

Each task writes files to `backend/project/work_dir/<task_id>/`:

- `paper.md`
- `paper.docx`
- `paper.pdf`
- `paper.json`
- `solution.ipynb`
- `solution.py`
- generated figures
- `status.json`
- `messages.jsonl`
- `request.json`

## Notes

- PDF generation depends on `pandoc` and at least one supported PDF engine such as `xelatex`.
- OpenAlex-based paper search is optional. Leave `OPENALEX_EMAIL` empty if you do not need it.

## Local Setup

### 1. Required runtime

- Python `>= 3.12`
- `uv`
- `pandoc`
- at least one PDF engine such as `xelatex`

### 2. Install dependencies

```bash
cd backend
uv sync --locked
```

This creates `.venv` and installs the packages declared in [pyproject.toml](./pyproject.toml), including:

- `fastapi[standard]`
- `litellm`
- `jupyter-client`
- `pypandoc-binary`

### 3. Configure models

Edit `.env.dev`.

For Zhipu OpenAI-compatible API, the repository default is:

```env
COORDINATOR_MODEL=openai/glm-4.7
COORDINATOR_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
MODELER_MODEL=openai/glm-4.7
MODELER_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
CODER_MODEL=openai/glm-4.7
CODER_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
WRITER_MODEL=openai/glm-4.7
WRITER_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
```

Fill the four `*_API_KEY` values before startup.

Notes:

- `E2B_API_KEY` is optional because the current workflow uses the local interpreter by default.
- The backend-only version stores task state in local files, so Redis is not required for local runs.

### 4. Start the server

```bash
cd backend
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --ws-ping-interval 60 --ws-ping-timeout 120
```

After startup:

- Swagger docs: `http://127.0.0.1:8000/docs`
- Status page: `http://127.0.0.1:8000/status`
