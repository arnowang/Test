#!/usr/bin/env python3
"""
CutPet 彩虹小院 · 本地零依赖后端服务
- 所有上传照片、训练好的 LoRA 权重、写实写真与视频均永久保存在本地 ./data/ 目录
- 自动代理转发到 Google Colab Pro GPU 服务，并在 Colab 重启时自动恢复本地备份的 LoRA
"""

import base64
import json
import mimetypes
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
LORAS_DIR = DATA_DIR / "loras"
OUTPUTS_DIR = DATA_DIR / "outputs"
DB_PATH = DATA_DIR / "db.json"

for d in (STATIC_DIR, DATA_DIR, UPLOADS_DIR, LORAS_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DB_LOCK = threading.Lock()
LOCAL_TASKS: Dict[str, Dict[str, Any]] = {}
LORA_SYNC_STATE: Dict[str, Dict[str, Any]] = {}


def download_lora_background(colab_url: str, pet_id: str) -> None:
    """后台流式分块下载 LoRA 到本地，不阻塞用户立刻在 Colab 生图/生视频"""
    cur = LORA_SYNC_STATE.get(pet_id, {})
    if cur.get("status") == "downloading":
        return
    LORA_SYNC_STATE[pet_id] = {
        "status": "downloading",
        "downloaded_mb": 0.0,
        "total_mb": 0.0,
        "speed_kbps": 0.0,
        "message": "正在后台同步 LoRA 到本地...",
    }

    def _worker():
        url = colab_url.rstrip("/") + f"/api/loras/{pet_id}"
        local_path = LORAS_DIR / f"{pet_id}.safetensors"
        part_path = LORAS_DIR / f"{pet_id}.safetensors.part"
        for attempt in range(1, 4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "CutPet-Local-Demo/1.0"})
                t0 = time.time()
                downloaded = 0
                with urllib.request.urlopen(req, timeout=60) as resp:
                    total = int(resp.headers.get("Content-Length") or 0)
                    total_mb = round(total / (1024 * 1024), 2) if total else 0.0
                    with part_path.open("wb") as f:
                        while True:
                            chunk = resp.read(262144)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            elapsed = max(time.time() - t0, 0.01)
                            speed_kbps = round((downloaded / 1024.0) / elapsed, 1)
                            dl_mb = round(downloaded / (1024 * 1024), 2)
                            LORA_SYNC_STATE[pet_id] = {
                                "status": "downloading",
                                "downloaded_mb": dl_mb,
                                "total_mb": total_mb,
                                "speed_kbps": speed_kbps,
                                "message": f"后台备份中 {dl_mb}/{total_mb or '?'} MB ({speed_kbps} KB/s)",
                            }
                part_path.replace(local_path)
                size_mb = round(local_path.stat().st_size / (1024 * 1024), 2)
                db = load_db()
                pet = find_pet(db, pet_id)
                if pet:
                    pet["lora_file"] = f"{pet_id}.safetensors"
                    pet["lora_size_mb"] = size_mb
                    pet["trained_on_colab"] = True
                    pet["trained_at"] = pet.get("trained_at") or time.strftime("%Y-%m-%d %H:%M:%S")
                    save_db(db)
                LORA_SYNC_STATE[pet_id] = {
                    "status": "synced",
                    "downloaded_mb": size_mb,
                    "total_mb": size_mb,
                    "speed_kbps": 0.0,
                    "message": f"已永久备份至本地 ({size_mb} MB)",
                }
                print(f"✅ [后台备份完成] {pet_id}.safetensors ({size_mb} MB)")
                return
            except Exception as e:
                print(f"⚠️ [后台备份重试 {attempt}/3] {pet_id}: {e}")
                time.sleep(2)
        LORA_SYNC_STATE[pet_id] = {
            "status": "error",
            "message": "后台下载较慢或中断（不影响当前直接在 Colab 生图，可稍后重试备份）",
        }

    threading.Thread(target=_worker, daemon=True).start()


def load_db() -> Dict[str, Any]:
    with DB_LOCK:
        if not DB_PATH.exists():
            return {"colab_url": "", "pets": [], "outputs": []}
        try:
            return json.loads(DB_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"colab_url": "", "pets": [], "outputs": []}


def save_db(db: Dict[str, Any]) -> None:
    with DB_LOCK:
        tmp = DB_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(DB_PATH)


def find_pet(db: Dict[str, Any], pet_id: str) -> Optional[Dict[str, Any]]:
    for p in db.get("pets", []):
        if p["id"] == pet_id:
            return p
    return None


def colab_request(
    base_url: str,
    path: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
) -> Any:
    url = base_url.rstrip("/") + path
    headers = {"User-Agent": "CutPet-Local-Demo/1.0"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            if "application/json" in ctype:
                return json.loads(raw.decode("utf-8"))
            return raw
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Colab HTTP {e.code}: {detail[:300]}")
    except Exception as e:
        raise RuntimeError(f"连接 Colab 失败 ({url}): {e}")


def ensure_lora_on_colab(base_url: str, pet_id: str, remote_loras: List[str]) -> bool:
    """如果本地已有 LoRA 权重但 Colab 重启后丢失，自动上传恢复"""
    if pet_id in remote_loras:
        return True
    local_lora = LORAS_DIR / f"{pet_id}.safetensors"
    if not local_lora.exists():
        return False
    b64 = base64.b64encode(local_lora.read_bytes()).decode("utf-8")
    colab_request(base_url, f"/api/loras/{pet_id}", method="POST", payload={"lora_base64": b64}, timeout=180)
    return True


def poll_colab_task_worker(
    local_task_id: str,
    colab_url: str,
    colab_task_id: str,
    task_type: str,
    meta: Dict[str, Any],
):
    """后台轮询 Colab 任务，并在完成后自动将 LoRA / 图片 / 视频保存到本地 data/ 目录"""
    try:
        while True:
            time.sleep(1.5)
            remote = colab_request(colab_url, f"/api/tasks/{colab_task_id}", method="GET", timeout=180)
            status = remote.get("status", "running")
            LOCAL_TASKS[local_task_id].update(
                {
                    "status": "saving" if status == "completed" else status,
                    "progress": min(int(remote.get("progress", 0)), 96) if status == "completed" else int(remote.get("progress", 0)),
                    "message": remote.get("message", "处理中..."),
                    "step": remote.get("step"),
                    "total_steps": remote.get("total_steps"),
                    "loss": remote.get("loss"),
                }
            )
            if status == "failed":
                LOCAL_TASKS[local_task_id]["error"] = remote.get("error") or remote.get("message")
                return

            if status == "completed":
                res = remote.get("result", {})
                if task_type == "train_lora":
                    pet_id = meta["pet_id"]
                    size_mb = res.get("size_mb", 0)
                    db = load_db()
                    pet = find_pet(db, pet_id)
                    if pet:
                        pet["lora_file"] = f"{pet_id}.safetensors"
                        pet["lora_size_mb"] = size_mb
                        pet["trained_on_colab"] = True
                        pet["trigger_token"] = res.get("trigger_token", f"sks_{pet_id[:8]}")
                        pet["trained_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                        pet["train_steps"] = res.get("steps", meta.get("steps", 400))
                        save_db(db)

                    # 改为默认秒存在 Google Drive，不再自动经慢速隧道拉回几十 MB LoRA 占用带宽
                    LOCAL_TASKS[local_task_id].update(
                        {
                            "status": "completed",
                            "progress": 100,
                            "message": f"✨ 训练完成！专属 LoRA ({size_mb} MB) 已秒存至 Google Drive，现可立即生成写实写真与视频！",
                            "result": {"pet_id": pet_id, "lora_file": f"{pet_id}.safetensors", "size_mb": size_mb},
                        }
                    )
                    return

                elif task_type == "generate_image":
                    LOCAL_TASKS[local_task_id]["message"] = "正在将写实写真保存到本地文件夹..."
                    filename = res.get("filename") or f"img_{uuid.uuid4().hex[:8]}.png"
                    raw_img = base64.b64decode(res["image_base64"])
                    out_path = OUTPUTS_DIR / filename
                    out_path.write_bytes(raw_img)

                    record = {
                        "id": uuid.uuid4().hex[:10],
                        "type": "image",
                        "pet_id": meta["pet_id"],
                        "second_pet_id": meta.get("second_pet_id"),
                        "scene": meta.get("scene_label") or meta.get("scene_prompt", ""),
                        "prompt_used": res.get("prompt_used", ""),
                        "lora_applied": res.get("lora_applied", False),
                        "seed": res.get("seed"),
                        "filename": filename,
                        "url": f"/data/outputs/{filename}",
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    db = load_db()
                    db["outputs"].insert(0, record)
                    save_db(db)

                    LOCAL_TASKS[local_task_id].update(
                        {
                            "status": "completed",
                            "progress": 100,
                            "message": f"写实写真已保存至本地 data/outputs/{filename}",
                            "result": record,
                        }
                    )
                    return

                elif task_type == "generate_video":
                    LOCAL_TASKS[local_task_id]["message"] = "正在将写实短视频保存到本地文件夹..."
                    filename = res.get("filename") or f"vid_{uuid.uuid4().hex[:8]}.mp4"
                    raw_vid = base64.b64decode(res["video_base64"])
                    out_path = OUTPUTS_DIR / filename
                    out_path.write_bytes(raw_vid)

                    record = {
                        "id": uuid.uuid4().hex[:10],
                        "type": "video",
                        "pet_id": meta["pet_id"],
                        "source_image_id": meta.get("source_image_id"),
                        "scene": meta.get("motion_prompt", "日常微动"),
                        "video_engine": res.get("video_engine", meta.get("video_engine", "cogvideox")),
                        "size_mb": round(len(raw_vid) / (1024 * 1024), 2),
                        "filename": filename,
                        "url": f"/data/outputs/{filename}",
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    db = load_db()
                    db["outputs"].insert(0, record)
                    save_db(db)

                    LOCAL_TASKS[local_task_id].update(
                        {
                            "status": "completed",
                            "progress": 100,
                            "message": f"写实视频已保存至本地 data/outputs/{filename}",
                            "result": record,
                        }
                    )
                    return
    except Exception as e:
        LOCAL_TASKS[local_task_id].update(
            {"status": "failed", "message": f"任务异常中断: {e}", "error": str(e)}
        )


class CutPetHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _serve_file(self, file_path: Path):
        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        ctype, _ = mimetypes.guess_type(str(file_path))
        if file_path.suffix.lower() == ".mp4":
            ctype = "video/mp4"
        ctype = ctype or "application/octet-stream"
        file_size = file_path.stat().st_size

        # 支持 HTTP Range 请求，保证浏览器 <video> 流畅播放与拖拽
        range_header = self.headers.get("Range")
        if range_header:
            m = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if m:
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else file_size - 1
                end = min(end, file_size - 1)
                length = end - start + 1
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.end_headers()
                with file_path.open("rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(file_size))
        self.end_headers()
        with file_path.open("rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/" or path == "/index.html":
            return self._serve_file(STATIC_DIR / "index.html")
        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            target = (STATIC_DIR / rel).resolve()
            if str(target).startswith(str(STATIC_DIR.resolve())):
                return self._serve_file(target)
            return self.send_error(HTTPStatus.FORBIDDEN)
        if path.startswith("/data/"):
            rel = path[len("/data/"):]
            target = (DATA_DIR / rel).resolve()
            if str(target).startswith(str(DATA_DIR.resolve())):
                return self._serve_file(target)
            return self.send_error(HTTPStatus.FORBIDDEN)

        if path == "/api/state":
            db = load_db()
            # 校验本地 LoRA 文件真实存在，并附带后台同步进度
            for p in db.get("pets", []):
                lora_f = p.get("lora_file") or f"{p['id']}.safetensors"
                local_exists = (LORAS_DIR / lora_f).exists()
                p["has_local_lora"] = local_exists
                if local_exists:
                    p["lora_file"] = lora_f
                    p["lora_size_mb"] = round((LORAS_DIR / lora_f).stat().st_size / (1024 * 1024), 2)
                p["lora_ready"] = bool(local_exists or p.get("trained_on_colab"))
                p["lora_sync"] = LORA_SYNC_STATE.get(p["id"])
            return self._send_json(
                {
                    "colab_url": db.get("colab_url", ""),
                    "pets": db.get("pets", []),
                    "outputs": db.get("outputs", []),
                    "storage_path": str(DATA_DIR),
                }
            )

        if path == "/api/colab/health":
            db = load_db()
            colab_url = db.get("colab_url", "").strip()
            if not colab_url:
                return self._send_json({"connected": False, "message": "尚未配置 Colab 公网地址"})
            try:
                info = colab_request(colab_url, "/health", method="GET", timeout=15)
                # 自动识别 Google Drive / Colab 上已训练好的 LoRA（不再自动拉回本地，零等待）
                remote_loras = set(info.get("available_loras") or [])
                changed = False
                for p in db.get("pets", []):
                    has_remote = p["id"] in remote_loras
                    if has_remote and not p.get("trained_on_colab"):
                        p["trained_on_colab"] = True
                        p["lora_file"] = f"{p['id']}.safetensors"
                        p["trained_at"] = p.get("trained_at") or time.strftime("%Y-%m-%d %H:%M:%S")
                        changed = True
                if changed:
                    save_db(db)
                return self._send_json({"connected": True, "info": info})
            except Exception as e:
                return self._send_json({"connected": False, "message": str(e)})

        if path.startswith("/api/tasks/"):
            task_id = path.split("/")[-1]
            task = LOCAL_TASKS.get(task_id)
            if not task:
                return self._send_json({"error": "任务不存在"}, status=404)
            return self._send_json(task)

        return self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            body = self._read_json()
        except Exception as e:
            return self._send_json({"error": f"无效 JSON: {e}"}, status=400)

        try:
            # 保存并测试 Colab 地址
            if path == "/api/config":
                colab_url = body.get("colab_url", "").strip().rstrip("/")
                db = load_db()
                db["colab_url"] = colab_url
                save_db(db)
                if not colab_url:
                    return self._send_json({"connected": False, "message": "已清空 Colab 地址"})
                try:
                    info = colab_request(colab_url, "/health", method="GET", timeout=15)
                    return self._send_json({"connected": True, "info": info})
                except Exception as e:
                    return self._send_json({"connected": False, "message": str(e)})

            # 新建宠物档案并持久化原始照片
            if path == "/api/pets":
                name = (body.get("name") or "").strip()
                if not name:
                    return self._send_json({"error": "请填写宠物名字"}, status=400)
                pet_id = f"pet_{uuid.uuid4().hex[:8]}"
                pet_dir = UPLOADS_DIR / pet_id
                pet_dir.mkdir(parents=True, exist_ok=True)

                saved_photos = []
                for idx, item in enumerate(body.get("photos") or []):
                    b64 = item.get("data", "")
                    orig_name = item.get("name", f"photo_{idx}.jpg")
                    ext = Path(orig_name).suffix.lower()
                    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
                        ext = ".jpg"
                    if "," in b64:
                        b64 = b64.split(",", 1)[1]
                    raw = base64.b64decode(b64)
                    fname = f"{idx:02d}_{uuid.uuid4().hex[:6]}{ext}"
                    (pet_dir / fname).write_bytes(raw)
                    saved_photos.append(f"/data/uploads/{pet_id}/{fname}")

                pet = {
                    "id": pet_id,
                    "name": name,
                    "breed": (body.get("breed") or "").strip(),
                    "features": (body.get("features") or "").strip(),
                    "personality": (body.get("personality") or "").strip(),
                    "photos": saved_photos,
                    "lora_file": None,
                    "lora_size_mb": 0,
                    "has_local_lora": False,
                    "trigger_token": f"sks_{pet_id[:8]}",
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                db = load_db()
                db["pets"].insert(0, pet)
                save_db(db)
                return self._send_json({"pet": pet})

            # 给已有宠物追加照片
            m_add = re.match(r"^/api/pets/([^/]+)/photos$", path)
            if m_add:
                pet_id = m_add.group(1)
                db = load_db()
                pet = find_pet(db, pet_id)
                if not pet:
                    return self._send_json({"error": "宠物档案不存在"}, status=404)
                pet_dir = UPLOADS_DIR / pet_id
                pet_dir.mkdir(parents=True, exist_ok=True)
                start_idx = len(pet.get("photos", []))
                for offset, item in enumerate(body.get("photos") or []):
                    b64 = item.get("data", "")
                    ext = Path(item.get("name", "photo.jpg")).suffix.lower()
                    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
                        ext = ".jpg"
                    if "," in b64:
                        b64 = b64.split(",", 1)[1]
                    raw = base64.b64decode(b64)
                    fname = f"{start_idx + offset:02d}_{uuid.uuid4().hex[:6]}{ext}"
                    (pet_dir / fname).write_bytes(raw)
                    pet["photos"].append(f"/data/uploads/{pet_id}/{fname}")
                save_db(db)
                return self._send_json({"pet": pet})

            # 可选：手动将 Google Drive 上的 LoRA 额外备份一份到本地
            m_sync = re.match(r"^/api/pets/([^/]+)/sync_lora$", path)
            if m_sync:
                pet_id = m_sync.group(1)
                db = load_db()
                colab_url = db.get("colab_url", "").strip()
                if not colab_url:
                    return self._send_json({"error": "请先连接 Google Colab"}, status=400)
                download_lora_background(colab_url, pet_id)
                return self._send_json({"status": "started", "pet_id": pet_id})

            # 发起 Colab LoRA 训练
            m_train = re.match(r"^/api/pets/([^/]+)/train$", path)
            if m_train:
                pet_id = m_train.group(1)
                db = load_db()
                colab_url = db.get("colab_url", "").strip()
                if not colab_url:
                    return self._send_json({"error": "请先在页面顶部连接 Google Colab Pro 地址"}, status=400)
                pet = find_pet(db, pet_id)
                if not pet:
                    return self._send_json({"error": "宠物档案不存在"}, status=404)

                pet_dir = UPLOADS_DIR / pet_id
                img_files = sorted(pet_dir.glob("*.*"))
                if not img_files:
                    return self._send_json({"error": "请至少上传 1 张宠物照片后再训练"}, status=400)

                images_b64 = [base64.b64encode(p.read_bytes()).decode("utf-8") for p in img_files[:15]]
                pet_desc = " ".join(filter(None, [pet.get("breed", ""), pet.get("features", "")]))
                steps = int(body.get("steps", 400))
                rank = int(body.get("rank", 16))

                remote_resp = colab_request(
                    colab_url,
                    "/api/train_lora",
                    method="POST",
                    payload={
                        "pet_id": pet_id,
                        "pet_name": pet["name"],
                        "pet_desc": pet_desc,
                        "images_base64": images_b64,
                        "steps": steps,
                        "rank": rank,
                    },
                    timeout=120,
                )
                local_tid = uuid.uuid4().hex[:12]
                LOCAL_TASKS[local_tid] = {
                    "task_id": local_tid,
                    "type": "train_lora",
                    "status": "running",
                    "progress": 2,
                    "message": "已提交到 Google Colab Pro GPU，正在启动专属 LoRA 训练...",
                }
                threading.Thread(
                    target=poll_colab_task_worker,
                    args=(local_tid, colab_url, remote_resp["task_id"], "train_lora", {"pet_id": pet_id, "steps": steps}),
                    daemon=True,
                ).start()
                return self._send_json({"task_id": local_tid})

            # 生成写实场景写真（单宠 / 双宠）
            if path == "/api/generate/image":
                db = load_db()
                colab_url = db.get("colab_url", "").strip()
                if not colab_url:
                    return self._send_json({"error": "请先连接 Google Colab Pro GPU 服务"}, status=400)
                pet_id = body.get("pet_id")
                pet = find_pet(db, pet_id)
                if not pet:
                    return self._send_json({"error": "请选择宠物"}, status=404)

                second_pet_id = body.get("second_pet_id") or None
                second_pet = find_pet(db, second_pet_id) if second_pet_id else None

                # 检查并自动恢复 Colab 上的 LoRA
                health = colab_request(colab_url, "/health", method="GET", timeout=20)
                remote_loras = health.get("available_loras", [])
                ensure_lora_on_colab(colab_url, pet_id, remote_loras)
                if second_pet_id:
                    ensure_lora_on_colab(colab_url, second_pet_id, remote_loras)

                pet_desc = " ".join(filter(None, [pet.get("breed", ""), pet.get("features", "")]))
                second_desc = (
                    " ".join(filter(None, [second_pet.get("breed", ""), second_pet.get("features", "")]))
                    if second_pet
                    else None
                )
                scene_prompt = body.get("scene_prompt", "sunlit green meadow lawn, happy relaxed pose")
                scene_label = body.get("scene_label", scene_prompt)

                remote_resp = colab_request(
                    colab_url,
                    "/api/generate_image",
                    method="POST",
                    payload={
                        "pet_id": pet_id,
                        "pet_desc": pet_desc,
                        "scene_prompt": scene_prompt,
                        "second_pet_id": second_pet_id,
                        "second_pet_desc": second_desc,
                        "lora_scale": float(body.get("lora_scale", 0.85)),
                        "num_steps": int(body.get("num_steps", 28)),
                        "seed": body.get("seed"),
                        "width": int(body.get("width", 1152)),
                        "height": int(body.get("height", 640)),
                    },
                    timeout=60,
                )
                local_tid = uuid.uuid4().hex[:12]
                LOCAL_TASKS[local_tid] = {
                    "task_id": local_tid,
                    "type": "generate_image",
                    "status": "running",
                    "progress": 5,
                    "message": "已提交写实写真任务到 Colab GPU...",
                }
                threading.Thread(
                    target=poll_colab_task_worker,
                    args=(
                        local_tid,
                        colab_url,
                        remote_resp["task_id"],
                        "generate_image",
                        {
                            "pet_id": pet_id,
                            "second_pet_id": second_pet_id,
                            "scene_prompt": scene_prompt,
                            "scene_label": scene_label,
                        },
                    ),
                    daemon=True,
                ).start()
                return self._send_json({"task_id": local_tid})

            # 图生视频（驱动小院动态）
            if path == "/api/generate/video":
                db = load_db()
                colab_url = db.get("colab_url", "").strip()
                if not colab_url:
                    return self._send_json({"error": "请先连接 Google Colab Pro GPU 服务"}, status=400)
                pet_id = body.get("pet_id")
                source_url = body.get("source_image_url", "")
                source_image_id = body.get("source_image_id")
                motion_prompt = body.get(
                    "motion_prompt", "在金色阳光下温柔地轻轻眨眼，自然舒缓呼吸，耳朵微微一动，微风轻拂毛发和草叶，镜头极其缓慢推进"
                )
                video_engine = body.get("video_engine", "wan2.2")

                if not source_url.startswith("/data/"):
                    return self._send_json({"error": "无效的首帧图片路径"}, status=400)
                src_path = (DATA_DIR / source_url[len("/data/"):]).resolve()
                if not src_path.exists():
                    return self._send_json({"error": "本地首帧图片文件不存在"}, status=404)

                img_b64 = base64.b64encode(src_path.read_bytes()).decode("utf-8")
                remote_resp = colab_request(
                    colab_url,
                    "/api/generate_video",
                    method="POST",
                    payload={
                        "pet_id": pet_id,
                        "image_base64": img_b64,
                        "motion_prompt": motion_prompt,
                        "video_engine": video_engine,
                        "num_steps": int(body.get("num_steps", 35)),
                    },
                    timeout=90,
                )
                local_tid = uuid.uuid4().hex[:12]
                LOCAL_TASKS[local_tid] = {
                    "task_id": local_tid,
                    "type": "generate_video",
                    "status": "running",
                    "progress": 5,
                    "message": "已提交写实动态视频任务到 Colab GPU...",
                }
                threading.Thread(
                    target=poll_colab_task_worker,
                    args=(
                        local_tid,
                        colab_url,
                        remote_resp["task_id"],
                        "generate_video",
                        {
                            "pet_id": pet_id,
                            "source_image_id": source_image_id,
                            "motion_prompt": motion_prompt,
                            "video_engine": video_engine,
                        },
                    ),
                    daemon=True,
                ).start()
                return self._send_json({"task_id": local_tid})

            return self._send_json({"error": "接口不存在"}, status=404)
        except Exception as e:
            return self._send_json({"error": str(e)}, status=500)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        m_pet = re.match(r"^/api/pets/([^/]+)$", path)
        if m_pet:
            pet_id = m_pet.group(1)
            db = load_db()
            colab_url = db.get("colab_url", "").strip()
            db["pets"] = [p for p in db.get("pets", []) if p["id"] != pet_id]
            db["outputs"] = [
                o for o in db.get("outputs", [])
                if o.get("pet_id") != pet_id and o.get("second_pet_id") != pet_id
            ]
            save_db(db)
            shutil.rmtree(UPLOADS_DIR / pet_id, ignore_errors=True)
            (LORAS_DIR / f"{pet_id}.safetensors").unlink(missing_ok=True)
            (LORAS_DIR / f"{pet_id}.safetensors.part").unlink(missing_ok=True)
            if colab_url:
                try:
                    colab_request(colab_url, f"/api/loras/{pet_id}", method="DELETE", timeout=15)
                except Exception:
                    pass
            return self._send_json({"deleted": pet_id})
        return self._send_json({"error": "Not found"}, status=404)


def main():
    port = int(os.environ.get("PORT", "8787"))
    server = ThreadingHTTPServer(("127.0.0.1", port), CutPetHandler)
    print("=" * 60)
    print(f"🌈 CutPet 彩虹小院 · 网页测试 Demo 已启动")
    print(f"👉 浏览器访问: http://127.0.0.1:{port}")
    print(f"📁 本地文件保存目录: {DATA_DIR}")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")


if __name__ == "__main__":
    main()
