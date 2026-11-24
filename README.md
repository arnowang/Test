# 🌈 CutPet 彩虹小院 · 写实宠物数字重生测试 Demo（路线 B MVP）

基于 **单宠专属写实 LoRA（RealVisXL V5.0） + 写实图生视频（CogVideoX-5B / Wan2.1-14B）** 的本地网页测试 Demo。

- **本地数据永久存储**：所有宠物照片、专属 LoRA 权重（`.safetensors`）、写实写真（`.png`）与写实短视频（`.mp4`）均自动保存在当前目录下的 `data/` 文件夹中。
- **Colab Pro 免费算力加速**：将 `colab/pet_lora_mvp_server.ipynb` 上传至 Google Colab Pro（推荐选择 **A100 GPU** 或 **L4 GPU**），运行后将打印的公网地址填入本网页顶部即可开始测试。

## 快速启动

```bash
cd /Users/arnowang/Downloads/cutpet
./start.sh
```

启动后在浏览器访问：http://127.0.0.1:8787

## 目录结构

- `app.py`：本地零依赖后端服务（文件持久化 + Colab API 代理 + LoRA 自动备份/恢复）
- `static/`：治愈系“彩虹小院”网页端界面（宠物建档、LoRA 唤醒、写实生图、图生视频、双宠合拍、本地画廊）
- `colab/pet_lora_mvp_server.ipynb`：一键上传到 Google Colab Pro 的 GPU 推理与训练服务笔记本
- `data/`：本地永久数据仓
  - `db.json`：宠物档案与作品记录
  - `uploads/`：上传的宠物原始照片
  - `loras/`：从 Colab 自动下载保存的单宠专属 LoRA 权重（`.safetensors`）
  - `outputs/`：生成的写实写真和写实动态小视频
