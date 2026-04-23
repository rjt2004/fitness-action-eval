<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import { onBeforeRouteLeave } from "vue-router";
import { ElMessage } from "element-plus";
import { getLiveSessionDetail, getLiveSessionPreviewFrame, startLiveSession, stopLiveSession } from "@/api/liveSession";
import { getPoseModelOptions } from "@/api/system";
import { getTemplateDetail, getTemplateList } from "@/api/template";

const templates = ref([]);
const selectedTemplate = ref(null);
const poseModelOptions = ref([]);
const templateLoading = ref(false);
const loading = ref(false);
const currentSession = ref(null);
const referenceVideoRef = ref(null);
const previewFrameUrl = ref("");
let timer = null;
let previewCounter = 0;
let leaveStopping = false;

const REALTIME_DEFAULTS = {
  camera_width: 480,
  camera_height: 270,
  camera_mirror: true,
  capture_error_frames: false,
  frame_stride: 1,
  smooth_window: 1,
  pose_model: "lite",
  hint_threshold: 0.2,
  hint_min_interval: 60,
  max_hints: 360,
  ref_search_window: 60,
};

const form = reactive({
  template_id: "",
  session_name: "实时跟练会话",
  camera_source: "0",
  ...REALTIME_DEFAULTS,
});

const runtimeModelOptions = [{ value: "follow_template", label: "跟随模板模型" }];
const sessionActive = computed(() => ["pending", "running"].includes(currentSession.value?.status || ""));
const realtimeInfo = computed(() => {
  const runtime = currentSession.value?.runtime_payload || {};
  const summary = currentSession.value?.summary_payload || {};
  const latestHint = Array.isArray(summary.hints) && summary.hints.length ? summary.hints[summary.hints.length - 1] : {};
  return {
    phase_name: runtime.phase_name || summary.final_phase_name || currentSession.value?.final_phase_name || "",
    part: runtime.part || latestHint.part || currentSession.value?.final_part || "",
    message: runtime.message || latestHint.message || "",
    score: runtime.score ?? currentSession.value?.avg_score ?? "",
    local_error: runtime.local_error ?? "",
  };
});

const realtimeMessage = computed(() => {
  if (!currentSession.value?.id) return "开始跟练后，这里会显示实时纠错提示";
  if (realtimeInfo.value.message) return realtimeInfo.value.message;
  if (sessionActive.value) return "当前动作保持较好，继续跟随参考视频";
  return "本次会话已结束";
});

function toMediaUrl(path) {
  if (!path) return "";
  const normalized = path.replaceAll("\\", "/");
  const index = normalized.indexOf("/media/");
  if (index >= 0) return normalized.slice(index);
  return normalized.startsWith("media/") ? `/${normalized}` : normalized;
}

function revokePreviewUrl() {
  if (previewFrameUrl.value) {
    URL.revokeObjectURL(previewFrameUrl.value);
    previewFrameUrl.value = "";
  }
}

function playReferenceVideo({ restart = false } = {}) {
  const el = referenceVideoRef.value;
  if (!el) return;
  if (restart) el.currentTime = 0;
  el.play().catch(() => {});
}

function pauseReferenceVideo() {
  referenceVideoRef.value?.pause();
}

function getStoredAccessToken() {
  try {
    const raw = window.localStorage.getItem("fitness_action_eval_auth");
    return raw ? JSON.parse(raw)?.accessToken || "" : "";
  } catch {
    return "";
  }
}

function sendStopKeepalive(sessionId) {
  const token = getStoredAccessToken();
  if (!token) return;
  fetch(`/api/live-session/sessions/${sessionId}/stop/`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    keepalive: true,
  }).catch(() => {});
}

async function loadTemplates() {
  const data = await getTemplateList();
  templates.value = data.filter((item) => item.status === "ready");
  if (!form.template_id && templates.value.length) {
    form.template_id = templates.value[0].id;
  }
}

async function loadPoseModelOptions() {
  poseModelOptions.value = await getPoseModelOptions();
}

async function loadTemplateDetail(templateId) {
  if (!templateId) {
    selectedTemplate.value = null;
    return;
  }
  templateLoading.value = true;
  try {
    selectedTemplate.value = await getTemplateDetail(templateId);
  } finally {
    templateLoading.value = false;
  }
}

function stopPolling() {
  if (timer) {
    window.clearInterval(timer);
    timer = null;
  }
}

async function refreshPreviewFrame(sessionId) {
  try {
    const blob = await getLiveSessionPreviewFrame(sessionId);
    if (!(blob instanceof Blob) || blob.size === 0) return;
    const nextUrl = URL.createObjectURL(blob);
    const prevUrl = previewFrameUrl.value;
    previewFrameUrl.value = nextUrl;
    if (prevUrl) URL.revokeObjectURL(prevUrl);
  } catch {
    // 会话启动初期预览帧可能暂时为空，忽略即可。
  }
}

function startPolling(sessionId) {
  stopPolling();
  previewCounter = 0;
  timer = window.setInterval(async () => {
    previewCounter += 1;
    await refreshPreviewFrame(sessionId);
    if (previewCounter % 4 === 0) {
      const data = await getLiveSessionDetail(sessionId);
      currentSession.value = data;
      if (["success", "failed", "stopped"].includes(data.status)) {
        pauseReferenceVideo();
        stopPolling();
      }
    }
  }, 250);
}

async function handleStart() {
  loading.value = true;
  revokePreviewUrl();
  try {
    const payload = {
      template_id: form.template_id,
      session_name: form.session_name,
      camera_source: String(Number(form.camera_source)) === form.camera_source ? Number(form.camera_source) : form.camera_source,
      camera_width: form.camera_width,
      camera_height: form.camera_height,
      camera_mirror: form.camera_mirror,
      preview: false,
      capture_error_frames: form.capture_error_frames,
      frame_stride: form.frame_stride,
      smooth_window: form.smooth_window,
      pose_model: form.pose_model,
      hint_threshold: form.hint_threshold,
      hint_min_interval: form.hint_min_interval,
      max_hints: form.max_hints,
      ref_search_window: form.ref_search_window,
    };
    const data = await startLiveSession(payload);
    currentSession.value = data;
    playReferenceVideo({ restart: true });
    ElMessage.success("实时会话已启动");
    startPolling(data.id);
  } finally {
    loading.value = false;
  }
}

async function handleStop() {
  if (!currentSession.value?.id) return;
  await stopLiveSession(currentSession.value.id);
  pauseReferenceVideo();
  ElMessage.success("已发送停止信号");
}

async function stopSessionBeforeLeave() {
  if (leaveStopping || !currentSession.value?.id || !sessionActive.value) return;
  leaveStopping = true;
  try {
    await stopLiveSession(currentSession.value.id);
  } catch {
    sendStopKeepalive(currentSession.value.id);
  } finally {
    pauseReferenceVideo();
    stopPolling();
    leaveStopping = false;
  }
}

function handlePageHide() {
  if (currentSession.value?.id && sessionActive.value) {
    sendStopKeepalive(currentSession.value.id);
  }
}

watch(
  () => form.template_id,
  (value) => {
    loadTemplateDetail(value);
  },
);

onMounted(() => {
  loadPoseModelOptions();
  loadTemplates();
  window.addEventListener("pagehide", handlePageHide);
});

onBeforeRouteLeave(async () => {
  await stopSessionBeforeLeave();
});

onBeforeUnmount(() => {
  handlePageHide();
  pauseReferenceVideo();
  stopPolling();
  revokePreviewUrl();
  window.removeEventListener("pagehide", handlePageHide);
});
</script>

<template>
  <div class="page-shell">
    <h2 class="page-title">实时跟练</h2>

    <div class="content-grid">
      <section class="soft-card page-panel reference-panel">
        <h3 class="section-title">参考视频</h3>
        <div v-loading="templateLoading">
          <video
            v-if="selectedTemplate?.source_video_path"
            ref="referenceVideoRef"
            class="preview-video reference-video"
            :src="toMediaUrl(selectedTemplate.source_video_path)"
            controls
            preload="metadata"
          />
          <el-empty v-else description="请选择可用模板" />
        </div>
      </section>

      <section class="soft-card page-panel params-panel">
        <h3 class="section-title">会话参数</h3>
        <el-form label-width="120px">
          <div class="grid-two">
            <div>
              <el-form-item label="会话名称">
                <el-input v-model="form.session_name" />
              </el-form-item>
              <el-form-item label="动作模板">
                <el-select v-model="form.template_id" style="width: 100%">
                  <el-option v-for="item in templates" :key="item.id" :label="item.template_name" :value="item.id" />
                </el-select>
              </el-form-item>
              <el-form-item label="姿态模型">
                <el-select v-model="form.pose_model" style="width: 100%">
                  <el-option
                    v-for="item in [...runtimeModelOptions, ...poseModelOptions]"
                    :key="item.value"
                    :label="item.label"
                    :value="item.value"
                  />
                </el-select>
              </el-form-item>
              <el-form-item label="摄像头源">
                <el-input v-model="form.camera_source" />
              </el-form-item>
              <el-form-item label="分辨率宽度">
                <el-input-number v-model="form.camera_width" :min="320" :max="1920" />
              </el-form-item>
              <el-form-item label="分辨率高度">
                <el-input-number v-model="form.camera_height" :min="240" :max="1080" />
              </el-form-item>
              <el-form-item label="镜像显示">
                <el-switch v-model="form.camera_mirror" />
              </el-form-item>
              <el-form-item label="保存错误帧">
                <el-switch v-model="form.capture_error_frames" />
              </el-form-item>
            </div>

            <div>
              <el-form-item label="抽帧步长">
                <el-input-number v-model="form.frame_stride" :min="1" :max="10" />
              </el-form-item>
              <el-form-item label="平滑窗口">
                <el-input-number v-model="form.smooth_window" :min="1" :max="15" />
              </el-form-item>
              <el-form-item label="提示阈值">
                <el-input-number v-model="form.hint_threshold" :min="0.01" :max="1" :step="0.01" />
              </el-form-item>
              <el-form-item label="提示间隔">
                <el-input-number v-model="form.hint_min_interval" :min="1" :max="100" />
              </el-form-item>
              <el-form-item label="提示上限">
                <el-input-number v-model="form.max_hints" :min="1" :max="9999" />
              </el-form-item>
              <el-form-item label="搜索窗口">
                <el-input-number v-model="form.ref_search_window" :min="10" :max="120" />
              </el-form-item>
            </div>
          </div>

          <el-form-item>
            <el-button type="primary" :loading="loading" @click="handleStart">开始跟练</el-button>
            <el-button type="danger" plain @click="handleStop">停止</el-button>
          </el-form-item>
        </el-form>
      </section>

      <section class="soft-card page-panel live-panel">
        <h3 class="section-title">实时画面</h3>
        <img v-if="previewFrameUrl" :src="previewFrameUrl" class="preview-video live-video" alt="实时预览" />
        <el-empty v-else description="开始跟练后，这里会显示实时画面" />
      </section>

      <section class="soft-card page-panel coach-panel">
        <h3 class="section-title">实时提示</h3>
        <div class="live-coach-card">
          <div class="live-coach-card__label">当前提示</div>
          <div class="live-coach-card__message">{{ realtimeMessage }}</div>
          <div class="live-coach-grid">
            <div class="live-coach-field">
              <span>动作阶段</span>
              <strong>{{ realtimeInfo.phase_name || "--" }}</strong>
            </div>
            <div class="live-coach-field">
              <span>关注部位</span>
              <strong>{{ realtimeInfo.part || "--" }}</strong>
            </div>
            <div class="live-coach-field">
              <span>当前分数</span>
              <strong>{{ realtimeInfo.score !== "" ? Number(realtimeInfo.score).toFixed(1) : "--" }}</strong>
            </div>
            <div class="live-coach-field">
              <span>局部偏差</span>
              <strong>{{ realtimeInfo.local_error !== "" ? Number(realtimeInfo.local_error).toFixed(3) : "--" }}</strong>
            </div>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.content-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.page-panel {
  padding: 22px;
}

.preview-video {
  display: block;
  width: 100%;
  border-radius: 16px;
  background: #000;
  object-fit: contain;
}

.reference-video {
  max-height: 300px;
}

.live-video {
  max-height: 360px;
}

.grid-two {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px 18px;
}

.live-coach-card {
  padding: 20px;
  border: 1px solid rgba(20, 83, 45, 0.14);
  border-radius: 20px;
  background:
    radial-gradient(circle at top left, rgba(187, 247, 208, 0.78), transparent 38%),
    linear-gradient(135deg, #f0fdf4 0%, #f8fafc 100%);
}

.live-coach-card__label {
  color: #166534;
  font-size: 15px;
  font-weight: 800;
  letter-spacing: 0.08em;
}

.live-coach-card__message {
  margin-top: 10px;
  color: #0f172a;
  font-size: 28px;
  font-weight: 900;
  line-height: 1.35;
}

.live-coach-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 18px;
}

.live-coach-field {
  min-height: 86px;
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.8);
  box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.06);
}

.live-coach-field span {
  display: block;
  color: #64748b;
  font-size: 14px;
  font-weight: 700;
}

.live-coach-field strong {
  display: block;
  margin-top: 8px;
  color: #0f172a;
  font-size: 21px;
  font-weight: 900;
  line-height: 1.45;
}

@media (max-width: 1200px) {
  .content-grid,
  .grid-two,
  .live-coach-grid {
    grid-template-columns: 1fr;
  }

  .live-coach-card__message {
    font-size: 24px;
  }
}
</style>
