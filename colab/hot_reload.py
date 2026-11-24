import os
import subprocess
import time
import urllib.request

RAW_URL = "https://raw.githubusercontent.com/arnowang/Test/cutpet-mvp/colab/server.py"
SERVER_PATH = "/content/cutpet_server/server.py"
os.makedirs("/content/cutpet_server", exist_ok=True)

print("⬇️ 正在从 GitHub 拉取最新 server.py ...")
req = urllib.request.Request(RAW_URL + f"?t={int(time.time())}", headers={"Cache-Control": "no-cache"})
code = urllib.request.urlopen(req, timeout=30).read()
with open(SERVER_PATH, "wb") as f:
    f.write(code)
print(f"✅ 已更新 {SERVER_PATH} ({len(code)} bytes)")

print("🔄 正在无缝重启 FastAPI（保留原 Cloudflare 公网地址与 Google Drive 挂载）...")
subprocess.run("pkill -f 'uvicorn server:app' || true", shell=True)
time.sleep(1)

lora_dir = "/content/drive/MyDrive/CutPet_Test/loras" if os.path.exists("/content/drive/MyDrive") else "/content/cutpet_server/loras"
env = os.environ.copy()
env["CUTPET_LORA_DIR"] = lora_dir

subprocess.Popen(
    ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"],
    cwd="/content/cutpet_server",
    env=env,
    stdout=open("/tmp/cutpet_uvicorn.log", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)

for _ in range(25):
    time.sleep(1)
    try:
        resp = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2).read().decode()
        print("✅ 热更新完成！当前服务状态:", resp)
        break
    except Exception:
        pass
else:
    print("❌ 重启超时，日志:", open("/tmp/cutpet_uvicorn.log").read())
    raise SystemExit(1)
