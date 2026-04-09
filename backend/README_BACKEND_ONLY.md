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
