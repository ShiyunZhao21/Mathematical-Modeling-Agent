from fastapi import APIRouter


router = APIRouter(tags=["system"])


@router.get("/")
async def root():
    return {
        "name": "MathModelAgent Backend",
        "mode": "backend-only",
        "message": "Service is running",
    }


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "backend",
    }
