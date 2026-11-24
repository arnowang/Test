import base64
import gc
import io
import os
import random
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageEnhance, ImageOps
from pydantic import BaseModel

# ============================================================
# 基础配置与目录（支持挂载 Google Drive 持久化 LoRA）
# ============================================================
BASE_DIR = Path(os.environ.get("CUTPET_WORKDIR", "/content/cutpet_server"))
DATASET_DIR = BASE_DIR / "datasets"
LORA_DIR = Path(os.environ.get("CUTPET_LORA_DIR", str(BASE_DIR / "loras")))
OUTPUT_DIR = BASE_DIR / "outputs"
for d in (DATASET_DIR, LORA_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

SDXL_MODEL_ID = os.environ.get("SDXL_MODEL_ID", "SG161222/RealVisXL_V5.0")
SDXL_FALLBACK_ID = "stabilityai/stable-diffusion-xl-base-1.0"
COGVIDEO_MODEL_ID = os.environ.get("COGVIDEO_MODEL_ID", "THUDM/CogVideoX-5b-I2V")
WAN_MODEL_ID = os.environ.get("WAN_MODEL_ID", "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")

app = FastAPI(title="CutPet Colab GPU Engine (Route B - Photorealistic)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 常见中文宠物与场景词汇自动补全，防止纯中文 Prompt 影响 CLIP 写实还原
CN_TO_EN_HINTS = {
    "金毛": "golden retriever dog",
    "拉布拉多": "labrador retriever dog",
    "柯基": "welsh corgi dog",
    "泰迪": "toy poodle dog",
    "比熊": "bichon frise dog",
    "博美": "pomeranian dog",
    "柴犬": "shiba inu dog",
    "边牧": "border collie dog",
    "萨摩耶": "samoyed dog",
    "哈士奇": "siberian husky dog",
    "法斗": "french bulldog",
    "雪纳瑞": "schnauzer dog",
    "田园犬": "chinese rural dog",
    "小狗": "cute dog",
    "狗": "dog",
    "三花猫": "calico cat with orange black and white patches",
    "三花": "calico cat",
    "奶牛猫": "tuxedo black and white cat",
    "橘猫": "orange tabby cat",
    "狸花猫": "brown tabby dragon li cat",
    "布偶猫": "ragdoll cat with blue eyes",
    "布偶": "ragdoll cat",
    "美短": "american shorthair tabby cat",
    "英短": "british shorthair cat",
    "蓝猫": "british shorthair blue grey cat",
    "银渐层": "silver shaded british shorthair cat",
    "金渐层": "golden shaded british shorthair cat",
    "暹罗": "siamese cat",
    "无毛猫": "sphynx cat",
    "小猫": "cute cat",
    "猫": "cat",
    "白手套": "white paws",
    "白袜子": "white socks on paws",
    "黑鼻头": "black nose",
    "粉鼻头": "pink nose",
    "绿眼睛": "green eyes",
    "蓝眼睛": "blue eyes",
    "琥珀眼": "amber eyes",
    "立耳": "upright ears",
    "折耳": "folded ears",
    "垂耳": "floppy ears",
    "短毛": "short fur",
    "长毛": "long fluffy fur",
    "草坪": "sunlit green meadow lawn",
    "海边": "golden sunset beach by the ocean",
    "窗台": "warm sunny wooden windowsill indoors",
    "樱花": "spring park pathway with falling pink cherry blossoms",
    "雪地": "soft winter snow field in warm morning light",
    "壁炉": "cozy living room rug beside a warm glowing fireplace",
    "奔跑": "happily running joyfully",
    "睡觉": "peacefully sleeping curled up",
    "打盹": "relaxing and napping peacefully",
    "摇尾巴": "gently wagging tail looking warmly at camera",
    "晒太阳": "basking in warm golden sunlight",
}


def enrich_prompt(text: str) -> str:
    if not text:
        return ""
    extras = []
    for cn, en in CN_TO_EN_HINTS.items():
        if cn in text and en not in extras:
            extras.append(en)
    # 保留原始 ASCII 文本，同时拼接自动映射的英文语义
    ascii_part = " ".join(
        part for part in text.replace("，", ",").replace("。", ",").split(",")
        if any(c.isascii() and c.isalpha() for c in part)
    ).strip()
    parts = [p for p in [ascii_part, ", ".join(extras)] if p]
    return ", ".join(parts) if parts else "adorable pet"


def decode_base64_image(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def encode_file_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def get_gpu_info() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False, "name": "CPU", "vram_gb": 0.0, "bf16_supported": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "vram_gb": round(props.total_memory / (1024 ** 3), 2),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }


# ============================================================
# GPU 显存互斥管理器（避免训练/生图/生视频 OOM）
# ============================================================
class GPUEngineManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_mode: Optional[str] = None
        self.pipe: Any = None
        self.loaded_loras: List[str] = []

    def unload(self):
        if self.pipe is not None:
            try:
                del self.pipe
            except Exception:
                pass
            self.pipe = None
        self.current_mode = None
        self.loaded_loras = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def get_sdxl_pipe(self):
        if self.current_mode == "sdxl" and self.pipe is not None:
            return self.pipe
        self.unload()
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

        dtype = torch.float16
        model_id = SDXL_MODEL_ID
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                variant="fp16",
            )
        except Exception:
            try:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                )
            except Exception:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    SDXL_FALLBACK_ID,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    variant="fp16",
                )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )
        pipe.to("cuda")
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        self.pipe = pipe
        self.current_mode = "sdxl"
        self.loaded_loras = []
        return pipe

    def get_video_pipe(self, engine: str = "cogvideox"):
        target_mode = f"video_{engine}"
        if self.current_mode == target_mode and self.pipe is not None:
            return self.pipe
        self.unload()
        gpu = get_gpu_info()
        dtype = torch.bfloat16 if gpu["bf16_supported"] else torch.float16

        if engine == "wan2.1":
            from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
            from transformers import CLIPVisionModel

            image_encoder = CLIPVisionModel.from_pretrained(
                WAN_MODEL_ID, subfolder="image_encoder", torch_dtype=torch.float32
            )
            vae = AutoencoderKLWan.from_pretrained(
                WAN_MODEL_ID, subfolder="vae", torch_dtype=torch.float32
            )
            pipe = WanImageToVideoPipeline.from_pretrained(
                WAN_MODEL_ID,
                vae=vae,
                image_encoder=image_encoder,
                torch_dtype=dtype,
            )
            pipe.enable_model_cpu_offload()
        else:
            from diffusers import CogVideoXDPMScheduler, CogVideoXImageToVideoPipeline

            pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                COGVIDEO_MODEL_ID,
                torch_dtype=dtype,
            )
            pipe.scheduler = CogVideoXDPMScheduler.from_config(
                pipe.scheduler.config, timestep_spacing="trailing"
            )
            pipe.enable_model_cpu_offload()
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()

        self.pipe = pipe
        self.current_mode = target_mode
        return pipe


engine_mgr = GPUEngineManager()
TASKS: Dict[str, Dict[str, Any]] = {}


def update_task(task_id: str, **kwargs):
    if task_id in TASKS:
        TASKS[task_id].update(kwargs)
        TASKS[task_id]["updated_at"] = time.time()


# ============================================================
# 核心模块 1：单宠专属写实 SDXL DreamBooth LoRA 极速训练
# ============================================================
def run_train_lora_task(
    task_id: str,
    pet_id: str,
    pet_name: str,
    pet_desc: str,
    images_b64: List[str],
    steps: int = 400,
    rank: int = 16,
    lr: float = 1e-4,
):
    try:
        update_task(task_id, status="running", progress=5, message="正在保存并预处理宠物训练照片...")
        pet_dataset_dir = DATASET_DIR / pet_id
        pet_dataset_dir.mkdir(parents=True, exist_ok=True)
        pil_images: List[Image.Image] = []
        for idx, b64_str in enumerate(images_b64):
            img = decode_base64_image(b64_str)
            # 居中裁剪并缩放到 1024x1024 保持构图不拉伸
            img_sq = ImageOps.fit(img, (1024, 1024), method=Image.Resampling.LANCZOS)
            img_sq.save(pet_dataset_dir / f"img_{idx:02d}.jpg", quality=95)
            pil_images.append(img_sq)

        if not pil_images:
            raise ValueError("未接收到有效的宠物照片")

        trigger_token = f"sks_{pet_id[:8]}"
        en_desc = enrich_prompt(pet_desc)
        instance_prompt = f"a realistic candid photo of {trigger_token}, {en_desc}, natural fur texture, sharp focus"

        with engine_mgr.lock:
            update_task(task_id, status="running", progress=12, message="正在加载写实底座模型（RealVisXL）并释放多余显存...")
            engine_mgr.unload()

            from diffusers import DDPMScheduler, StableDiffusionXLPipeline
            from diffusers.utils import convert_state_dict_to_diffusers
            from peft import LoraConfig, get_peft_model
            from peft.utils import get_peft_model_state_dict

            gpu = get_gpu_info()
            use_bf16 = gpu["bf16_supported"]
            weight_dtype = torch.bfloat16 if use_bf16 else torch.float16

            try:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    SDXL_MODEL_ID, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
                )
            except Exception:
                try:
                    pipe = StableDiffusionXLPipeline.from_pretrained(
                        SDXL_MODEL_ID, torch_dtype=torch.float16, use_safetensors=True
                    )
                except Exception:
                    pipe = StableDiffusionXLPipeline.from_pretrained(
                        SDXL_FALLBACK_ID, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
                    )

            # 1) 预先编码 Text Embeddings，随后立刻卸载双文本编码器，极大省显存
            update_task(task_id, progress=20, message="正在编码宠物专属视觉特征与文本提示...")
            pipe.text_encoder.to("cuda")
            pipe.text_encoder_2.to("cuda")
            with torch.no_grad():
                prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                    prompt=instance_prompt,
                    device="cuda",
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                prompt_embeds = prompt_embeds.to(dtype=weight_dtype, device="cuda")
                pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype, device="cuda")

            pipe.text_encoder.to("cpu")
            pipe.text_encoder_2.to("cpu")
            del pipe.text_encoder, pipe.text_encoder_2
            gc.collect()
            torch.cuda.empty_cache()

            # 2) 预先用 float32 VAE 编码所有训练图（含水平翻转数据增强）为 Latents
            update_task(task_id, progress=28, message=f"正在编码 {len(pil_images)} 张宠物照片（含多视角增强）...")
            vae = pipe.vae.to(device="cuda", dtype=torch.float32)
            vae.requires_grad_(False)
            vae_scale = vae.config.scaling_factor

            cached_latents = []
            with torch.no_grad():
                for img in pil_images:
                    variants = [img, ImageOps.mirror(img)]
                    if len(pil_images) <= 4:
                        variants.append(ImageEnhance.Brightness(img).enhance(1.06))
                        variants.append(ImageEnhance.Brightness(ImageOps.mirror(img)).enhance(0.95))
                    for v_img in variants:
                        arr = np.array(v_img, dtype=np.float32) / 127.5 - 1.0
                        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float32)
                        lat = vae.encode(tensor).latent_dist.sample() * vae_scale
                        cached_latents.append(lat.to(dtype=weight_dtype).detach())

            del vae, pipe.vae
            gc.collect()
            torch.cuda.empty_cache()

            # 3) 配置 UNet + LoRA 训练
            unet = pipe.unet.to(device="cuda", dtype=weight_dtype)
            unet.requires_grad_(False)
            unet.enable_gradient_checkpointing()

            lora_config = LoraConfig(
                r=rank,
                lora_alpha=rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            unet = get_peft_model(unet, lora_config)
            for p in unet.parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)

            trainable_params = [p for p in unet.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)
            noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
            add_time_ids = torch.tensor(
                [[1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0]],
                device="cuda",
                dtype=weight_dtype,
            )
            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

            scaler = torch.amp.GradScaler("cuda", enabled=(not use_bf16))
            unet.train()

            update_task(task_id, progress=32, message=f"开始训练专属写实 LoRA（共 {steps} 步）...")
            loss_history = []
            t0 = time.time()

            for step in range(1, steps + 1):
                latents = random.choice(cached_latents)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (1,), device="cuda"
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=weight_dtype):
                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                if use_bf16:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                loss_val = float(loss.detach().cpu().item())
                loss_history.append(loss_val)

                if step % 15 == 0 or step == steps:
                    avg_loss = sum(loss_history[-20:]) / len(loss_history[-20:])
                    pct = 32 + int(60 * (step / steps))
                    elapsed = int(time.time() - t0)
                    update_task(
                        task_id,
                        progress=pct,
                        step=step,
                        total_steps=steps,
                        loss=round(avg_loss, 4),
                        message=f"正在深度记忆 {pet_name} 的毛发与神态... ({step}/{steps} 步, loss={avg_loss:.4f}, 已耗时 {elapsed}s)",
                    )

            # 4) 导出标准 Diffusers / ComfyUI 通用 .safetensors LoRA 权重（转 fp16 体积减半，下载快 2 倍）
            update_task(task_id, progress=95, message="训练完成，正在导出轻量 fp16 .safetensors 权重...")
            raw_lora_sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
            unet_lora_state_dict = {
                k: (v.to(torch.float16) if torch.is_floating_point(v) else v)
                for k, v in raw_lora_sd.items()
            }
            lora_filename = f"{pet_id}.safetensors"
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=str(LORA_DIR),
                unet_lora_layers=unet_lora_state_dict,
                weight_name=lora_filename,
            )

            del unet, optimizer, cached_latents, pipe
            gc.collect()
            torch.cuda.empty_cache()

            lora_path = LORA_DIR / lora_filename
            size_mb = round(lora_path.stat().st_size / (1024 * 1024), 2)
            update_task(
                task_id,
                status="completed",
                progress=100,
                message=f"专属写实 LoRA 训练完成！（权重 {size_mb} MB，总耗时 {int(time.time() - t0)}s）",
                result={
                    "pet_id": pet_id,
                    "trigger_token": trigger_token,
                    "lora_filename": lora_filename,
                    "size_mb": size_mb,
                    "steps": steps,
                    "final_loss": round(sum(loss_history[-20:]) / len(loss_history[-20:]), 4),
                },
            )
    except Exception as e:
        traceback.print_exc()
        update_task(task_id, status="failed", message=f"训练失败: {str(e)}", error=traceback.format_exc())


# ============================================================
# 核心模块 2：单宠 / 双宠写实场景生图（挂载专属 LoRA）
# ============================================================
def run_generate_image_task(
    task_id: str,
    pet_id: str,
    pet_desc: str,
    scene_prompt: str,
    second_pet_id: Optional[str] = None,
    second_pet_desc: Optional[str] = None,
    lora_scale: float = 0.85,
    num_steps: int = 28,
    seed: Optional[int] = None,
):
    try:
        with engine_mgr.lock:
            update_task(task_id, status="running", progress=15, message="正在加载写实摄影引擎（RealVisXL）...")
            pipe = engine_mgr.get_sdxl_pipe()

            # 清理旧 adapter
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass

            lora1 = LORA_DIR / f"{pet_id}.safetensors"
            adapters = []
            weights = []
            trigger1 = f"sks_{pet_id[:8]}"
            en_desc1 = enrich_prompt(pet_desc)

            if lora1.exists():
                pipe.load_lora_weights(str(LORA_DIR), weight_name=lora1.name, adapter_name="pet1")
                adapters.append("pet1")

            if second_pet_id:
                lora2 = LORA_DIR / f"{second_pet_id}.safetensors"
                trigger2 = f"sks_{second_pet_id[:8]}"
                en_desc2 = enrich_prompt(second_pet_desc or "")
                if lora2.exists():
                    pipe.load_lora_weights(str(LORA_DIR), weight_name=lora2.name, adapter_name="pet2")
                    adapters.append("pet2")
                if len(adapters) == 2:
                    weights = [min(lora_scale, 0.72), min(lora_scale, 0.72)]
                elif len(adapters) == 1:
                    weights = [lora_scale]
                en_scene = enrich_prompt(scene_prompt)
                full_prompt = (
                    f"award-winning candid wildlife photograph of two adorable pets playing happily together side by side: "
                    f"on the left is {trigger1} ({en_desc1}), and on the right is {trigger2} ({en_desc2}), "
                    f"{en_scene}, warm golden hour natural sunlight, 85mm lens, shallow depth of field, "
                    f"ultra realistic fur detail, expressive warm eyes, 8k photorealistic"
                )
            else:
                if adapters:
                    weights = [lora_scale]
                en_scene = enrich_prompt(scene_prompt)
                full_prompt = (
                    f"award-winning candid photograph of {trigger1}, {en_desc1}, {en_scene}, "
                    f"warm golden hour natural sunlight, 85mm f/1.8 lens, soft bokeh background, "
                    f"ultra realistic fur texture, lifelike warm eyes, natural pose, 8k raw photo"
                )

            if adapters:
                pipe.set_adapters(adapters, adapter_weights=weights)

            negative_prompt = (
                "cartoon, anime, 3d render, plastic, doll, painting, illustration, blurry, lowres, "
                "extra legs, extra tail, two heads, deformed paws, mutated anatomy, fused bodies, "
                "bad proportions, watermark, text, logo, oversaturated, creepy eyes"
            )

            used_seed = seed if (seed is not None and seed >= 0) else random.randint(1, 2_000_000_000)
            generator = torch.Generator(device="cuda").manual_seed(used_seed)

            update_task(task_id, progress=45, message="正在挂载专属 LoRA 渲染 1024×1024 写实画面...")
            result_img = pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=5.5,
                width=1024,
                height=1024,
                generator=generator,
            ).images[0]

            out_name = f"img_{pet_id[:8]}_{int(time.time())}_{used_seed % 10000}.png"
            out_path = OUTPUT_DIR / out_name
            result_img.save(out_path, format="PNG")

            update_task(
                task_id,
                status="completed",
                progress=100,
                message="写实场景写真生成完成！",
                result={
                    "filename": out_name,
                    "seed": used_seed,
                    "prompt_used": full_prompt,
                    "lora_applied": bool(adapters),
                    "image_base64": encode_file_base64(out_path),
                },
            )
    except Exception as e:
        traceback.print_exc()
        update_task(task_id, status="failed", message=f"生图失败: {str(e)}", error=traceback.format_exc())


# ============================================================
# 核心模块 3：写实图生视频（CogVideoX-5B / Wan2.1-14B）
# ============================================================
def run_generate_video_task(
    task_id: str,
    pet_id: str,
    image_b64: str,
    motion_prompt: str,
    video_engine: str = "cogvideox",
    num_steps: int = 28,
):
    try:
        from diffusers.utils import export_to_video

        with engine_mgr.lock:
            engine_label = "Wan2.1-14B 旗舰写实引擎" if video_engine == "wan2.1" else "CogVideoX-5B 极速写实引擎"
            update_task(
                task_id,
                status="running",
                progress=12,
                message=f"正在切换显存并加载 {engine_label}（首次加载需下载权重，请耐心等待）...",
            )
            pipe = engine_mgr.get_video_pipe(video_engine)
            src_img = decode_base64_image(image_b64)
            en_motion = enrich_prompt(motion_prompt)
            full_motion_prompt = (
                f"Photorealistic natural cinematic video, {en_motion}, smooth gentle natural animal movement, "
                f"soft breeze moving fur gently, warm golden sunlight, stable camera, high fidelity, no distortion"
            )

            update_task(task_id, progress=40, message=f"正在使用 {engine_label} 渲染写实动态帧...")
            t0 = time.time()

            if video_engine == "wan2.1":
                input_img = ImageOps.fit(src_img, (832, 480), method=Image.Resampling.LANCZOS)
                output = pipe(
                    image=input_img,
                    prompt=full_motion_prompt,
                    negative_prompt="blurry, distorted, extra limbs, morphing, low quality, flickering, cartoon",
                    height=480,
                    width=832,
                    num_frames=49,
                    num_inference_steps=num_steps,
                    guidance_scale=5.0,
                ).frames[0]
                fps = 16
            else:
                input_img = ImageOps.fit(src_img, (720, 480), method=Image.Resampling.LANCZOS)
                output = pipe(
                    image=input_img,
                    prompt=full_motion_prompt,
                    num_videos_per_prompt=1,
                    num_inference_steps=num_steps,
                    num_frames=49,
                    guidance_scale=6.0,
                ).frames[0]
                fps = 8

            update_task(task_id, progress=92, message="正在封装浏览器兼容 H.264 MP4 视频...")
            out_name = f"vid_{pet_id[:8]}_{int(time.time())}.mp4"
            raw_path = OUTPUT_DIR / f"raw_{out_name}"
            out_path = OUTPUT_DIR / out_name
            export_to_video(output, str(raw_path), fps=fps)

            import subprocess
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", str(raw_path),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "20", "-movflags", "+faststart", str(out_path),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                raw_path.unlink(missing_ok=True)
            except Exception:
                raw_path.rename(out_path)

            size_mb = round(out_path.stat().st_size / (1024 * 1024), 2)
            update_task(
                task_id,
                status="completed",
                progress=100,
                message=f"写实动态视频生成完成！（耗时 {int(time.time() - t0)}s, {size_mb} MB）",
                result={
                    "filename": out_name,
                    "size_mb": size_mb,
                    "fps": fps,
                    "video_engine": video_engine,
                    "video_base64": encode_file_base64(out_path),
                },
            )
    except Exception as e:
        traceback.print_exc()
        update_task(task_id, status="failed", message=f"视频生成失败: {str(e)}", error=traceback.format_exc())


# ============================================================
# FastAPI 接口定义
# ============================================================
class TrainLoraRequest(BaseModel):
    pet_id: str
    pet_name: str
    pet_desc: str = ""
    images_base64: List[str]
    steps: int = 400
    rank: int = 16
    lr: float = 1e-4


class UploadLoraRequest(BaseModel):
    lora_base64: str


class GenerateImageRequest(BaseModel):
    pet_id: str
    pet_desc: str = ""
    scene_prompt: str
    second_pet_id: Optional[str] = None
    second_pet_desc: Optional[str] = None
    lora_scale: float = 0.85
    num_steps: int = 28
    seed: Optional[int] = None


class GenerateVideoRequest(BaseModel):
    pet_id: str
    image_base64: str
    motion_prompt: str = "gently wagging tail, blinking softly, breathing peacefully in warm sunlight"
    video_engine: str = "cogvideox"
    num_steps: int = 28


@app.get("/health")
def health():
    loras = [p.stem for p in LORA_DIR.glob("*.safetensors")]
    is_drive = str(LORA_DIR).startswith("/content/drive")
    return {
        "status": "ok",
        "gpu": get_gpu_info(),
        "sdxl_model": SDXL_MODEL_ID,
        "current_mode": engine_mgr.current_mode,
        "available_loras": loras,
        "lora_storage": f"Google Drive ({LORA_DIR})" if is_drive else f"Colab 临时盘 ({LORA_DIR})",
        "drive_mounted": is_drive,
    }


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.post("/api/train_lora")
def api_train_lora(req: TrainLoraRequest):
    task_id = uuid.uuid4().hex[:12]
    TASKS[task_id] = {
        "task_id": task_id,
        "type": "train_lora",
        "status": "queued",
        "progress": 0,
        "message": "任务已加入 GPU 队列...",
        "created_at": time.time(),
    }
    t = threading.Thread(
        target=run_train_lora_task,
        args=(task_id, req.pet_id, req.pet_name, req.pet_desc, req.images_base64, req.steps, req.rank, req.lr),
        daemon=True,
    )
    t.start()
    return {"task_id": task_id}


@app.get("/api/loras/{pet_id}")
def download_lora(pet_id: str):
    path = LORA_DIR / f"{pet_id}.safetensors"
    if not path.exists():
        raise HTTPException(status_code=404, detail="LoRA not found on Colab")
    return FileResponse(str(path), filename=path.name, media_type="application/octet-stream")


@app.post("/api/loras/{pet_id}")
def upload_lora(pet_id: str, req: UploadLoraRequest):
    raw = base64.b64decode(req.lora_base64)
    path = LORA_DIR / f"{pet_id}.safetensors"
    path.write_bytes(raw)
    return {"status": "restored", "pet_id": pet_id, "size_mb": round(len(raw) / (1024 * 1024), 2)}


@app.delete("/api/loras/{pet_id}")
def delete_lora(pet_id: str):
    path = LORA_DIR / f"{pet_id}.safetensors"
    existed = path.exists()
    path.unlink(missing_ok=True)
    return {"status": "deleted", "pet_id": pet_id, "existed": existed}


@app.post("/api/generate_image")
def api_generate_image(req: GenerateImageRequest):
    task_id = uuid.uuid4().hex[:12]
    TASKS[task_id] = {
        "task_id": task_id,
        "type": "generate_image",
        "status": "queued",
        "progress": 0,
        "message": "生图任务已加入队列...",
        "created_at": time.time(),
    }
    t = threading.Thread(
        target=run_generate_image_task,
        args=(
            task_id,
            req.pet_id,
            req.pet_desc,
            req.scene_prompt,
            req.second_pet_id,
            req.second_pet_desc,
            req.lora_scale,
            req.num_steps,
            req.seed,
        ),
        daemon=True,
    )
    t.start()
    return {"task_id": task_id}


@app.post("/api/generate_video")
def api_generate_video(req: GenerateVideoRequest):
    task_id = uuid.uuid4().hex[:12]
    TASKS[task_id] = {
        "task_id": task_id,
        "type": "generate_video",
        "status": "queued",
        "progress": 0,
        "message": "视频生成任务已加入队列...",
        "created_at": time.time(),
    }
    t = threading.Thread(
        target=run_generate_video_task,
        args=(task_id, req.pet_id, req.image_base64, req.motion_prompt, req.video_engine, req.num_steps),
        daemon=True,
    )
    t.start()
    return {"task_id": task_id}
