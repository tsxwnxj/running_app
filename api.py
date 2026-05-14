"""FastAPI 백엔드 서버.

실행: python api.py
기본 포트: 8000
"""

import os
import uuid
import shutil
import asyncio
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="러닝 자세 분석 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# 처리 중인 작업 상태 저장 (job_id → status)
_jobs: dict[str, dict] = {}


@app.get("/")
def root():
    return {"status": "ok", "service": "러닝 자세 분석 API"}


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    view_angle: str = Form("side"),
):
    """영상 업로드 → 분석 시작 → job_id 반환."""
    job_id = uuid.uuid4().hex[:12]
    suffix = Path(video.filename).suffix or ".mp4"
    input_path  = RESULTS_DIR / f"{job_id}_input{suffix}"
    output_path = RESULTS_DIR / f"{job_id}_output.mp4"

    # 업로드 파일 저장
    with open(input_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    _jobs[job_id] = {"status": "processing", "progress": 0}

    # 비동기로 분석 시작
    asyncio.create_task(_run_analysis(job_id, str(input_path), str(output_path), view_angle))

    return {"job_id": job_id, "status": "processing"}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    """분석 진행 상태 조회."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job_id를 찾을 수 없습니다.")
    return job


@app.get("/result/{job_id}")
def get_result(job_id: str):
    """분석 완료된 결과 반환."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job_id를 찾을 수 없습니다.")
    if job["status"] != "done":
        raise HTTPException(202, "아직 처리 중입니다.")
    return job["result"]


@app.get("/video/{filename}")
def get_video(filename: str):
    """분석 완료된 영상 파일 반환."""
    path = RESULTS_DIR / filename
    if not path.exists():
        raise HTTPException(404, "파일을 찾을 수 없습니다.")
    return FileResponse(str(path), media_type="video/mp4")


async def _run_analysis(job_id: str, input_path: str, output_path: str, view_angle: str):
    try:
        from video_processor import process_video

        def progress_cb(current, total):
            pct = int(current / max(total, 1) * 100)
            _jobs[job_id]["progress"] = pct

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: process_video(input_path, output_path, view_angle, progress_cb),
        )

        video_filename = Path(output_path).name
        result["video_url"] = f"/video/{video_filename}"
        result["job_id"] = job_id

        _jobs[job_id] = {"status": "done", "progress": 100, "result": result}

        # 입력 파일 삭제 (용량 절약)
        try:
            os.remove(input_path)
        except OSError:
            pass

    except Exception as e:
        _jobs[job_id] = {"status": "error", "error": str(e)}
        raise


if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"

    print(f"\n{'='*50}")
    print(f"  러닝 자세 분석 API 서버 시작")
    print(f"  로컬:    http://127.0.0.1:8000")
    print(f"  네트워크: http://{local_ip}:8000")
    print(f"  같은 Wi-Fi의 스마트폰에서 위 네트워크 주소를 사용하세요.")
    print(f"{'='*50}\n")

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
