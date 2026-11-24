const SCENE_PRESETS = [
  {
    label: "🌿 阳光草坪",
    prompt: "full body resting happily on a lush sunlit green meadow lawn, wildflowers around, warm golden afternoon rim light, wide cinematic framing",
    motion: "在金色阳光下温柔地轻轻眨眼，嘴角自然舒缓呼吸，耳朵微微抖动了一下，微风轻拂毛发和周围野花草叶，镜头极其缓慢平缓推进",
  },
  {
    label: "🌅 落日海滩",
    prompt: "sitting peacefully on a golden sandy beach at sunset, gentle ocean waves in soft bokeh background, warm sunset glow on fur, wide cinematic shot",
    motion: "温柔地看向镜头轻轻眨眼，金色夕阳微风轻拂毛发，身后柔和的海浪缓慢流动，自然安静呼吸",
  },
  {
    label: "☀️ 午后暖阳窗台",
    prompt: "relaxing peacefully on a warm cozy wooden windowsill bathed in golden afternoon sunlight, soft white curtains, peaceful home atmosphere",
    motion: "在午后暖阳中惬意地慢慢眯起眼睛，胸口均匀舒缓呼吸，微风吹动窗帘光影轻轻晃动",
  },
  {
    label: "🌸 樱花小径",
    prompt: "sitting happily on a spring park pathway under blooming pink cherry blossom trees, soft petals floating in the air, wide natural portrait",
    motion: "粉色樱花花瓣在微风中缓缓飘落，温柔地轻轻眨眼微微抬头，毛发随微风轻轻拂动",
  },
  {
    label: "❄️ 暖阳雪地",
    prompt: "resting joyfully in soft fresh white winter snow under warm morning sunshine, crisp realistic fur with tiny snowflakes, natural candid photo",
    motion: "温暖晨光洒在毛发上，轻轻眨眼自然呼吸，耳朵微微一动，背景细雪缓慢飘落",
  },
  {
    label: "🔥 温暖壁炉小窝",
    prompt: "curled up comfortably on a soft cream rug beside a warm glowing fireplace in a cozy cabin living room, warm ambient light",
    motion: "在温暖跳动的壁炉火光映照下安稳舒缓地呼吸打盹，耳朵轻轻动了一下，氛围温馨宁静",
  },
];

let state = {
  colab_url: "",
  colab_connected: false,
  pets: [],
  outputs: [],
  selectedPetId: null,
  isDualMode: false,
  currentStageImage: null, // { id, url, scene }
  galleryFilter: "all",
  pendingNewPhotos: [],
  activeTaskTimer: null,
  busy: false,
};

const $ = (id) => document.getElementById(id);

async function apiFetch(url, options = {}) {
  const res = await fetch(url, options);
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || `请求失败 (${res.status})`);
  }
  return data;
}

// 前端适度缩放图片至最大 1400px，兼顾清晰度与上传速度
function fileToResizedBase64(file, maxDim = 1400) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        let { width, height } = img;
        if (width > maxDim || height > maxDim) {
          const ratio = Math.min(maxDim / width, maxDim / height);
          width = Math.round(width * ratio);
          height = Math.round(height * ratio);
        }
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, width, height);
        resolve({
          name: file.name,
          data: canvas.toDataURL("image/jpeg", 0.92),
        });
      };
      img.onerror = reject;
      img.src = reader.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function loadState() {
  const data = await apiFetch("/api/state");
  state.colab_url = data.colab_url || "";
  state.pets = data.pets || [];
  state.outputs = data.outputs || [];
  $("storagePathText").textContent = data.storage_path || "/Users/arnowang/Downloads/cutpet/data";
  if (state.colab_url && !$("colabUrlInput").value) {
    $("colabUrlInput").value = state.colab_url;
  }

  if (!state.selectedPetId && state.pets.length > 0) {
    state.selectedPetId = state.pets[0].id;
  } else if (state.selectedPetId && !state.pets.some((p) => p.id === state.selectedPetId)) {
    state.selectedPetId = state.pets[0]?.id || null;
  }

  renderPetList();
  renderWorkspace();
}

function formatColabStatus(info) {
  const gpu = info?.gpu;
  const gpuText = gpu?.available ? `${gpu.name} (${gpu.vram_gb}GB)` : "CPU 模式";
  const driveTag = info?.drive_mounted ? " · ☁️ Drive 已挂载" : " · 临时盘";
  return `已连接 · ${gpuText}${driveTag}`;
}

async function checkColabHealth(silent = true) {
  const dot = $("colabStatusDot");
  const txt = $("colabStatusText");
  try {
    const res = await apiFetch("/api/colab/health");
    if (res.connected) {
      state.colab_connected = true;
      dot.className = "status-dot online";
      txt.textContent = formatColabStatus(res.info);
      await loadState();
    } else {
      state.colab_connected = false;
      dot.className = "status-dot offline";
      txt.textContent = "未连接";
      if (!silent) alert("连接失败：" + res.message);
    }
  } catch (e) {
    state.colab_connected = false;
    dot.className = "status-dot offline";
    txt.textContent = "未连接";
    if (!silent) alert("连接失败：" + e.message);
  }
}

function getSelectedPet() {
  return state.pets.find((p) => p.id === state.selectedPetId) || null;
}

function renderPetList() {
  const box = $("petList");
  if (!state.pets.length) {
    box.innerHTML = `<div class="empty-hint">还没有毛孩子档案，点击上方「+ 新建档案」上传照片开始测试吧。</div>`;
    return;
  }
  box.innerHTML = state.pets
    .map((pet) => {
      const avatar = pet.photos?.[0] || "";
      const active = pet.id === state.selectedPetId ? "active" : "";
      const status = pet.lora_ready
        ? `✅ 已唤醒 (Drive 秒存)`
        : `📷 ${pet.photos?.length || 0} 张照 · 待唤醒`;
      return `
        <div class="pet-item ${active}" data-pet-id="${pet.id}">
          <img src="${avatar}" alt="${pet.name}" />
          <div class="pet-item-info">
            <div class="pet-item-name">${escapeHtml(pet.name)}</div>
            <div class="pet-item-sub">${escapeHtml(pet.breed || "萌宠")} · ${status}</div>
          </div>
        </div>
      `;
    })
    .join("");

  box.querySelectorAll(".pet-item").forEach((el) => {
    el.addEventListener("click", () => {
      state.selectedPetId = el.dataset.petId;
      state.currentStageImage = null;
      renderPetList();
      renderWorkspace();
    });
  });
}

function renderWorkspace() {
  const pet = getSelectedPet();
  if (!pet) {
    $("emptyWorkspace").classList.remove("hidden");
    $("petWorkspace").classList.add("hidden");
    return;
  }
  $("emptyWorkspace").classList.add("hidden");
  $("petWorkspace").classList.remove("hidden");

  $("heroAvatar").src = pet.photos?.[0] || "";
  $("heroName").textContent = pet.name;
  $("heroMeta").textContent = [
    pet.breed ? `品种：${pet.breed}` : "",
    pet.personality ? `性格：${pet.personality}` : "",
    `已存照片：${pet.photos?.length || 0} 张`,
  ]
    .filter(Boolean)
    .join(" · ");
  $("heroFeatures").textContent = pet.features ? `外貌特征：${pet.features}` : "";

  const badge = $("heroLoraBadge");
  const loraInfo = $("loraLocalInfo");
  if (pet.lora_ready) {
    badge.className = "badge badge-ok";
    badge.textContent = pet.has_local_lora
      ? `✨ 已唤醒（Drive + 本地 ${pet.lora_size_mb} MB）`
      : "✨ 已唤醒（Google Drive 已保存）";
    loraInfo.classList.remove("hidden");
    const syncHtml = pet.has_local_lora
      ? `本地副本：<code>data/loras/${pet.lora_file}</code>`
      : pet.lora_sync?.status === "downloading"
      ? `正在额外备份到本地：${escapeHtml(pet.lora_sync.message)}`
      : `<button id="btnManualSyncLora" class="btn btn-soft btn-sm" style="margin-left:8px;">⬇️ 额外备份一份到本地</button>`;
    loraInfo.innerHTML = `✅ 专属写实 LoRA 已保存在 Google Drive（完成时间：${pet.trained_at || ""}），<strong>现在即可直接前往「2. 写实工坊」生成写实写真与视频！</strong> ${syncHtml}`;
    const btnSync = $("btnManualSyncLora");
    if (btnSync) {
      btnSync.addEventListener("click", async () => {
        btnSync.disabled = true;
        btnSync.textContent = "已加入后台备份...";
        await apiFetch(`/api/pets/${pet.id}/sync_lora`, { method: "POST" });
        await loadState();
      });
    }
  } else {
    badge.className = "badge badge-warn";
    badge.textContent = "尚未唤醒专属 LoRA";
    loraInfo.classList.add("hidden");
  }

  // 原始照片网格
  $("petPhotosGrid").innerHTML = (pet.photos || [])
    .map((u) => `<img src="${u}" alt="宠物原图" />`)
    .join("");

  // 更新双宠选择器
  const others = state.pets.filter((p) => p.id !== pet.id);
  const sel = $("secondPetSelect");
  if (others.length === 0) {
    sel.innerHTML = `<option value="">请先在左侧再创建第 2 只宠物档案</option>`;
  } else {
    sel.innerHTML = others
      .map(
        (o) =>
          `<option value="${o.id}">${escapeHtml(o.name)}（${escapeHtml(o.breed || "萌宠")}${
            o.has_local_lora ? " · 已唤醒" : " · 未唤醒"
          }）</option>`
      )
      .join("");
  }

  // 更新小院舞台与画廊
  renderGallery();
  restoreLatestStageForPet(pet.id);
}

function restoreLatestStageForPet(petId) {
  const petOutputs = state.outputs.filter((o) => o.pet_id === petId || o.second_pet_id === petId);
  if (!petOutputs.length) {
    $("stagePlaceholder").classList.remove("hidden");
    $("stageImage").classList.add("hidden");
    $("stageVideo").classList.add("hidden");
    $("btnGenVideo").disabled = true;
    $("stageCaption").textContent = "生成写真或视频后将在此展示";
    return;
  }
  const latest = petOutputs[0];
  showInStage(latest);
}

function showInStage(item) {
  $("stagePlaceholder").classList.add("hidden");
  if (item.type === "video") {
    $("stageImage").classList.add("hidden");
    const vid = $("stageVideo");
    vid.src = item.url;
    vid.classList.remove("hidden");
    vid.play().catch(() => {});
    $("stageCaption").textContent = `写实视频 · ${item.scene} (${item.created_at})`;
  } else {
    const vid = $("stageVideo");
    vid.pause();
    vid.classList.add("hidden");
    const img = $("stageImage");
    img.src = item.url;
    img.classList.remove("hidden");
    state.currentStageImage = item;
    $("btnGenVideo").disabled = false;
    $("stageCaption").textContent = `写实写真 · ${item.scene} (${item.created_at})`;
  }
}

function renderGallery() {
  const pet = getSelectedPet();
  const grid = $("galleryGrid");
  if (!pet) return;
  let items = state.outputs.filter((o) => o.pet_id === pet.id || o.second_pet_id === pet.id);
  if (state.galleryFilter !== "all") {
    items = items.filter((o) => o.type === state.galleryFilter);
  }
  if (!items.length) {
    grid.innerHTML = `<div class="empty-hint">暂无生成作品，去「2. 写实工坊」创作第一张写实写真或视频吧。</div>`;
    return;
  }
  grid.innerHTML = items
    .map((item) => {
      const mediaHtml =
        item.type === "video"
          ? `<video class="gallery-thumb" src="${item.url}" muted loop playsinline onmouseover="this.play()" onmouseout="this.pause()"></video>`
          : `<img class="gallery-thumb" src="${item.url}" alt="作品" />`;
      const toVidBtn =
        item.type === "image"
          ? `<button class="btn btn-accent btn-sm btn-to-video" data-id="${item.id}">转视频</button>`
          : "";
      return `
        <div class="gallery-card">
          ${mediaHtml}
          <div class="gallery-body">
            <div class="gallery-title">${item.type === "video" ? "🎬" : "📸"} ${escapeHtml(item.scene || "")}</div>
            <div class="muted">${item.created_at}</div>
            <div class="gallery-actions">
              <button class="btn btn-soft btn-sm btn-preview" data-id="${item.id}">小院展示</button>
              ${toVidBtn}
              <a class="btn btn-soft btn-sm" href="${item.url}" download="${item.filename}">下载</a>
            </div>
          </div>
        </div>
      `;
    })
    .join("");

  grid.querySelectorAll(".btn-preview").forEach((btn) => {
    btn.addEventListener("click", () => {
      const item = state.outputs.find((o) => o.id === btn.dataset.id);
      if (item) {
        switchTab("tabStudio");
        showInStage(item);
      }
    });
  });

  grid.querySelectorAll(".btn-to-video").forEach((btn) => {
    btn.addEventListener("click", () => {
      const item = state.outputs.find((o) => o.id === btn.dataset.id);
      if (item) {
        switchTab("tabStudio");
        showInStage(item);
        triggerGenerateVideo();
      }
    });
  });
}

function setBusy(flag) {
  state.busy = flag;
  $("btnStartTrain").disabled = flag;
  $("btnGenImage").disabled = flag;
  if (flag) {
    $("btnGenVideo").disabled = true;
  } else if (state.currentStageImage) {
    $("btnGenVideo").disabled = false;
  }
}

// 统一任务进度轮询（防并发闪烁）
async function trackTask(taskId, title, onDone) {
  if (state.activeTaskTimer) {
    clearInterval(state.activeTaskTimer);
    state.activeTaskTimer = null;
  }
  setBusy(true);
  const banner = $("taskBanner");
  banner.classList.remove("hidden");
  $("taskTitle").textContent = title;
  $("taskPercent").textContent = "2%";
  $("taskProgressBar").style.width = "2%";
  $("taskMessage").textContent = "已提交到 Google Colab Pro GPU...";

  state.activeTaskTimer = setInterval(async () => {
    try {
      const task = await apiFetch(`/api/tasks/${taskId}`);
      const pct = task.progress || 0;
      $("taskPercent").textContent = `${pct}%`;
      $("taskProgressBar").style.width = `${pct}%`;
      $("taskMessage").textContent = task.message || "处理中...";

      if (task.status === "completed") {
        clearInterval(state.activeTaskTimer);
        state.activeTaskTimer = null;
        setBusy(false);
        await loadState();
        if (onDone) onDone(task.result);
        setTimeout(() => banner.classList.add("hidden"), 5000);
      } else if (task.status === "failed") {
        clearInterval(state.activeTaskTimer);
        state.activeTaskTimer = null;
        setBusy(false);
        $("taskTitle").textContent = "❌ 任务执行失败";
        $("taskMessage").textContent = task.message || task.error || "未知错误";
      }
    } catch (e) {
      clearInterval(state.activeTaskTimer);
      state.activeTaskTimer = null;
      setBusy(false);
      $("taskMessage").textContent = "轮询状态异常：" + e.message;
    }
  }, 1500);
}

function switchTab(tabId) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === tabId));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === tabId));
}

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function triggerGenerateVideo() {
  const pet = getSelectedPet();
  if (!pet || !state.currentStageImage) {
    alert("请先生成或选中一张写实写真作为首帧！");
    return;
  }
  if (!state.colab_connected) {
    alert("请先在页面顶部连接 Google Colab Pro GPU 地址！");
    return;
  }
  try {
    const res = await apiFetch("/api/generate/video", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pet_id: pet.id,
        source_image_id: state.currentStageImage.id,
        source_image_url: state.currentStageImage.url,
        motion_prompt: $("motionPromptInput").value.trim(),
        video_engine: $("videoEngineSelect").value,
      }),
    });
    trackTask(res.task_id, `🎬 正在为 ${pet.name} 生成写实动态小视频...`, (record) => {
      if (record) showInStage(record);
    });
  } catch (e) {
    alert(e.message);
  }
}

// ============================================================
// 初始化事件绑定
// ============================================================
function init() {
  // 场景预设按钮
  const presetBox = $("scenePresets");
  presetBox.innerHTML = SCENE_PRESETS.map(
    (s, idx) => `<button class="chip ${idx === 0 ? "active" : ""}" data-idx="${idx}">${s.label}</button>`
  ).join("");
  $("scenePromptInput").value = SCENE_PRESETS[0].prompt;

  presetBox.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      presetBox.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      const preset = SCENE_PRESETS[Number(chip.dataset.idx)];
      $("scenePromptInput").value = preset.prompt;
      $("motionPromptInput").value = preset.motion;
    });
  });

  // 连接 Colab
  $("btnConnectColab").addEventListener("click", async () => {
    const url = $("colabUrlInput").value.trim();
    if (!url) {
      alert("请先粘贴 Colab Notebook 输出的 https://xxxx.trycloudflare.com 地址");
      return;
    }
    $("colabStatusText").textContent = "连接测试中...";
    try {
      const res = await apiFetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ colab_url: url }),
      });
      if (res.connected) {
        state.colab_connected = true;
        $("colabStatusDot").className = "status-dot online";
        $("colabStatusText").textContent = formatColabStatus(res.info);
        await loadState();
      } else {
        state.colab_connected = false;
        $("colabStatusDot").className = "status-dot offline";
        $("colabStatusText").textContent = "连接失败";
        alert("无法连接到 Colab 服务：" + res.message);
      }
    } catch (e) {
      alert(e.message);
    }
  });

  // Tabs
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  });

  // 单宠 / 双宠切换
  $("modeSingleBtn").addEventListener("click", () => {
    state.isDualMode = false;
    $("modeSingleBtn").classList.add("active");
    $("modeDualBtn").classList.remove("active");
    $("secondPetBox").classList.add("hidden");
  });
  $("modeDualBtn").addEventListener("click", () => {
    state.isDualMode = true;
    $("modeDualBtn").classList.add("active");
    $("modeSingleBtn").classList.remove("active");
    $("secondPetBox").classList.remove("hidden");
  });

  // LoRA scale 显示
  $("loraScaleInput").addEventListener("input", (e) => {
    $("loraScaleVal").textContent = e.target.value;
  });

  // 新建宠物弹窗
  const openModal = () => {
    state.pendingNewPhotos = [];
    $("newPetName").value = "";
    $("newPetBreed").value = "";
    $("newPetFeatures").value = "";
    $("newPetPersonality").value = "";
    $("newPetPreviewGrid").innerHTML = "";
    $("modalNewPet").classList.remove("hidden");
  };
  const closeModal = () => $("modalNewPet").classList.add("hidden");
  $("btnOpenNewPet").addEventListener("click", openModal);
  $("btnWelcomeCreate").addEventListener("click", openModal);
  $("btnCloseModal").addEventListener("click", closeModal);
  $("btnCancelModal").addEventListener("click", closeModal);

  const handleNewFiles = async (files) => {
    for (const file of files) {
      if (!file.type.startsWith("image/")) continue;
      const item = await fileToResizedBase64(file);
      state.pendingNewPhotos.push(item);
    }
    $("newPetPreviewGrid").innerHTML = state.pendingNewPhotos
      .map((p) => `<img src="${p.data}" alt="${escapeHtml(p.name)}" />`)
      .join("");
  };

  $("dropZone").addEventListener("click", () => $("newPetPhotosInput").click());
  $("newPetPhotosInput").addEventListener("change", (e) => handleNewFiles(e.target.files));
  $("dropZone").addEventListener("dragover", (e) => e.preventDefault());
  $("dropZone").addEventListener("drop", (e) => {
    e.preventDefault();
    handleNewFiles(e.dataTransfer.files);
  });

  $("btnSubmitNewPet").addEventListener("click", async () => {
    const name = $("newPetName").value.trim();
    if (!name) return alert("请填写毛孩子昵称");
    if (!state.pendingNewPhotos.length) return alert("请至少上传 1 张毛孩子照片");
    try {
      const res = await apiFetch("/api/pets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          breed: $("newPetBreed").value.trim(),
          features: $("newPetFeatures").value.trim(),
          personality: $("newPetPersonality").value.trim(),
          photos: state.pendingNewPhotos,
        }),
      });
      closeModal();
      state.selectedPetId = res.pet.id;
      await loadState();
    } catch (e) {
      alert(e.message);
    }
  });

  // 追加照片
  $("inputAppendPhotos").addEventListener("change", async (e) => {
    const pet = getSelectedPet();
    if (!pet || !e.target.files.length) return;
    const photos = [];
    for (const f of e.target.files) {
      if (f.type.startsWith("image/")) photos.push(await fileToResizedBase64(f));
    }
    if (!photos.length) return;
    await apiFetch(`/api/pets/${pet.id}/photos`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ photos }),
    });
    e.target.value = "";
    await loadState();
  });

  // 删除宠物
  $("btnDeletePet").addEventListener("click", async () => {
    const pet = getSelectedPet();
    if (!pet) return;
    if (!confirm(`确定删除「${pet.name}」的测试档案和本地 LoRA 吗？`)) return;
    await apiFetch(`/api/pets/${pet.id}`, { method: "DELETE" });
    state.selectedPetId = null;
    await loadState();
  });

  // 开始 LoRA 训练
  $("btnStartTrain").addEventListener("click", async () => {
    const pet = getSelectedPet();
    if (!pet) return;
    if (!state.colab_connected) {
      alert("请先在页面顶部连接 Google Colab Pro GPU 地址！");
      return;
    }
    try {
      const res = await apiFetch(`/api/pets/${pet.id}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          steps: Number($("trainStepsSelect").value),
          rank: Number($("trainRankSelect").value),
        }),
      });
      trackTask(res.task_id, `✨ 正在为 ${pet.name} 训练专属写实 LoRA...`);
    } catch (e) {
      alert(e.message);
    }
  });

  // 生成写实场景写真
  $("btnGenImage").addEventListener("click", async () => {
    const pet = getSelectedPet();
    if (!pet) return;
    if (!state.colab_connected) {
      alert("请先在页面顶部连接 Google Colab Pro GPU 地址！");
      return;
    }
    const secondPetId = state.isDualMode ? $("secondPetSelect").value : null;
    if (state.isDualMode && !secondPetId) {
      alert("双宠合拍需要先在左侧创建第 2 只宠物档案哦！");
      return;
    }
    const activeChip = document.querySelector("#scenePresets .chip.active");
    const sceneLabel = activeChip ? activeChip.textContent : "自定义写实场景";
    const seedVal = $("seedInput").value.trim();
    const [wStr, hStr] = ($("aspectRatioSelect").value || "1152x640").split("x");

    try {
      const res = await apiFetch("/api/generate/image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pet_id: pet.id,
          second_pet_id: secondPetId || null,
          scene_prompt: $("scenePromptInput").value.trim(),
          scene_label: secondPetId ? `双宠合拍 · ${sceneLabel}` : sceneLabel,
          lora_scale: Number($("loraScaleInput").value),
          seed: seedVal ? Number(seedVal) : null,
          width: Number(wStr) || 1152,
          height: Number(hStr) || 640,
        }),
      });
      trackTask(res.task_id, `📸 正在渲染 ${pet.name} 的写实写真...`, (record) => {
        if (record) showInStage(record);
      });
    } catch (e) {
      alert(e.message);
    }
  });

  // 生成写实视频
  $("btnGenVideo").addEventListener("click", triggerGenerateVideo);

  // 小院互动小彩蛋
  const interactCopy = {
    pet: { emoji: "💛", text: "它开心地蹭了蹭你的手心～" },
    treat: { emoji: "🍖", text: "它摇着尾巴吃掉了最爱的小零食！" },
    flower: { emoji: "🌼", text: "彩虹小院里又多了一朵温暖的小花～" },
  };
  document.querySelectorAll("[data-interact]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const cfg = interactCopy[btn.dataset.interact];
      $("interactToast").textContent = cfg.text;
      const layer = $("floatHeartLayer");
      for (let i = 0; i < 4; i++) {
        const el = document.createElement("span");
        el.className = "float-emoji";
        el.textContent = cfg.emoji;
        el.style.left = `${20 + Math.random() * 60}%`;
        el.style.animationDelay = `${i * 0.12}s`;
        layer.appendChild(el);
        setTimeout(() => el.remove(), 1800);
      }
    });
  });

  // 画廊筛选
  document.querySelectorAll("[data-filter]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-filter]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.galleryFilter = btn.dataset.filter;
      renderGallery();
    });
  });

  loadState().then(() => checkColabHealth(true));

  // 如果有 LoRA 正在后台静默下载，每 3 秒更新一次备份进度，不干扰用户操作
  setInterval(() => {
    const downloading = state.pets.some((p) => p.lora_sync && p.lora_sync.status === "downloading");
    if (downloading && !state.busy) {
      loadState().catch(() => {});
    }
  }, 3000);
}

init();
