from fastapi import APIRouter
from fastapi.responses import HTMLResponse


router = APIRouter(tags=["系统"])


@router.get("/")
async def root():
    return {
        "name": "MathModelAgent 后端服务",
        "mode": "backend-only",
        "message": "服务运行中",
        "status_page": "/status",
        "docs": "/docs",
    }


@router.get("/status", response_class=HTMLResponse, include_in_schema=False)
async def status_page():
    return """
    <!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>MathModelAgent 后端状态</title>
      <style>
        body { font-family: 'Microsoft YaHei', Arial, sans-serif; background:#f7f8fa; color:#1f2328; margin:0; }
        .wrap { max-width: 900px; margin: 48px auto; background:#fff; border-radius:16px; padding:32px; box-shadow:0 8px 24px rgba(0,0,0,.08); }
        h1 { margin-top:0; }
        .ok { display:inline-block; padding:6px 12px; background:#e8f5e9; color:#1b5e20; border-radius:999px; font-weight:700; }
        .grid { display:grid; gap:16px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-top:24px; }
        .card { border:1px solid #e5e7eb; border-radius:12px; padding:16px; background:#fafafa; }
        a { color:#2563eb; text-decoration:none; }
        a:hover { text-decoration:underline; }
        code { background:#f1f5f9; padding:2px 6px; border-radius:6px; }
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="ok">服务运行中</div>
        <h1>MathModelAgent 后端状态页</h1>
        <p>这是后端服务的中文状态页面。你可以从这里进入接口文档、健康检查和静态产物目录。</p>
        <div class="grid">
          <div class="card">
            <h3>接口文档</h3>
            <p><a href="/docs" target="_blank">打开 Swagger 文档</a></p>
            <p><a href="/redoc" target="_blank">打开 ReDoc 文档</a></p>
          </div>
          <div class="card">
            <h3>健康检查</h3>
            <p><a href="/health" target="_blank">查看健康状态 JSON</a></p>
          </div>
          <div class="card">
            <h3>静态产物</h3>
            <p>任务产物将挂载在 <code>/static</code> 路径下。</p>
          </div>
        </div>
      </div>
    </body>
    </html>
    """


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "backend",
        "message": "后端服务正常",
    }
