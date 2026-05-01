<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import { onBeforeRouteLeave } from "vue-router";
import { ElMessage } from "element-plus";
import {
  getLiveSessions,
  getLiveSessionDetail,
  getLiveSessionPreviewFrame,
  startLiveSession,
  stopLiveSession,
} from "@/api/liveSession";
import { getPoseModelOptions } from "@/api/system";
import { getTemplateDetail, getTemplateList } from "@/api/template";

const templates = ref([]);
const selectedTemplate = ref(null);
const poseModelOptions = ref([]);
const templateLoading = ref(false);
const loading = ref(false);
const currentSession = ref(null);
const referenceVideoRef = ref(null);
const practiceAudioRef = ref(null);
const previewFrameUrl = ref("");
let timer = null;
let previewCounter = 0;
let leaveStopping = false;
let polling = false;

const REALTIME_DEFAULTS = {
  camera_width: 320,
  camera_height: 180,
  camera_mirror: false,
  capture_error_frames: false,
  frame_stride: 2,
  smooth_window: 1,
  pose_model: "lite",
  hint_threshold: 0.2,
  hint_min_interval: 60,
  max_hints: 240,
  ref_search_window: 12,
};

const form = reactive({
  template_id: "",
  session_name: "实时跟练会话",
  camera_source: "0",
  ...REALTIME_DEFAULTS,
});

const runtimeModelOptions = [{ value: "follow_template", label: "跟随模板模型" }];
const practiceAudioUrl = "/media/template_center/source/八段锦完整版.m4a";
const sessionActive = computed(() => ["pending", "running"].includes(currentSession.value?.status || ""));

const realtimeInfo = computed(() => {
  const runtime = currentSession.value?.runtime_payload || {};
  const summary = currentSession.value?.summary_payload || {};
  const latestHint = Array.isArray(summary.hints) && summary.hints.length ? summary.hints[summary.hints.length - 1] : {};
  const confidence = runtime.confidence || summary.runtime_state?.confidence || {};
  return {
    phase_name: runtime.phase_name || summary.final_phase_name || currentSession.value?.final_phase_name || "",
    part: runtime.part || latestHint.part || currentSession.value?.final_part || "",
    message: runtime.message || latestHint.message || "",
    local_error: runtime.local_error ?? "",
    confidence_mean: confidence.mean ?? "",
    confidence_min: confidence.min ?? "",
    valid_points: confidence.valid_points ?? "",
    total_points: confidence.total_points ?? "",
  };
});

const confidenceLegend = [
  { label: "高", color: "#50ff78", range: ">= 0.75" },
  { label: "中", color: "#ffdc00", range: "0.40 - 0.75" },
  { label: "低", color: "#ff5046", range: "< 0.40" },
];

const realtimeMessage = computed(() => {
  if (!currentSession.value?.id) return "开始跟练后，这里会显示实时纠错提示。";
  if (realtimeInfo.value.message) return realtimeInfo.value.message;
  if (sessionActive.value) return "当前动作保持较好，请继续跟随参考视频。";
  return "本次会话已结束。";
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
  const audio = practiceAudioRef.value;
  if (!el) return;
  if (restart) {
    el.currentTime = 0;
    if (audio) audio.currentTime = 0;
  }
  el.play().catch(() => {});
  audio?.play().catch(() => {});
}

function pauseReferenceVideo() {
  referenceVideoRef.value?.pause();
  practiceAudioRef.value?.pause();
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
    const realtimeTemplate =
      templates.value.find((item) => item.pose_model === "lite") ||
      templates.value.find((item) => String(item.template_name || "").toLowerCase().includes("lite")) ||
      templates.value[0];
    form.template_id = realtimeTemplate.id;
  }
}

async function loadLatestSession() {
  const sessions = await getLiveSessions();
  if (!Array.isArray(sessions) || !sessions.length) return;
  const latest = [...sessions].sort((a, b) => {
    const aTime = new Date(a.created_at || 0).getTime();
    const bTime = new Date(b.created_at || 0).getTime();
    return bTime - aTime;
  })[0];
  if (!latest?.id) return;
  currentSession.value = await getLiveSessionDetail(latest.id);
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
    window.clearTimeout(timer);
    timer = null;
  }
  polling = false;
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
    // 会话刚启动时可能暂时没有预览帧，这里直接忽略。
  }
}

async function pollSession(sessionId) {
  if (polling) return;
  polling = true;
  try {
    previewCounter += 1;
    await refreshPreviewFrame(sessionId);
    if (previewCounter % 3 === 0) {
      const data = await getLiveSessionDetail(sessionId);
      currentSession.value = data;
      if (["success", "failed", "stopped"].includes(data.status)) {
        pauseReferenceVideo();
        stopPolling();
        return;
      }
    }
  } finally {
    polling = false;
    if (timer !== null && currentSession.value?.id === sessionId && sessionActive.value) {
      timer = window.setTimeout(() => {
        pollSession(sessionId);
      }, 400);
    }
  }
}

function startPolling(sessionId) {
  stopPolling();
  previewCounter = 0;
  timer = window.setTimeout(() => {
    pollSession(sessionId);
  }, 0);
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
  loadLatestSession();
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
    <div class="content-grid">
      <section class="soft-card page-panel reference-panel">
        <h3 class="section-title">参考视频</h3>
        <div v-loading="templateLoading">
          <audio ref="practiceAudioRef" :src="practiceAudioUrl" preload="auto" />
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
                <el-input-number v-model="form.camera_height" :min="180" :max="1080" />
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
                <el-input-number v-model="form.ref_search_window" :min="6" :max="120" />
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
              <span>局部偏差</span>
              <strong>{{ realtimeInfo.local_error !== "" ? Number(realtimeInfo.local_error).toFixed(3) : "--" }}</strong>
            </div>
            <div class="live-coach-field">
              <span>识别置信度</span>
              <strong>{{ realtimeInfo.confidence_mean !== "" ? Number(realtimeInfo.confidence_mean).toFixed(2) : "--" }}</strong>
            </div>
            <div class="live-coach-field">
              <span>有效关键点</span>
              <strong>
                {{
                  realtimeInfo.valid_points !== "" && realtimeInfo.total_points !== ""
                    ? `${realtimeInfo.valid_points}/${realtimeInfo.total_points}`
                    : "--"
                }}
              </strong>
            </div>
          </div>
          <div class="confidence-legend">
            <span>骨架颜色</span>
            <div v-for="item in confidenceLegend" :key="item.label" class="confidence-legend__item">
              <i :style="{ backgroundColor: item.color }"></i>
              <strong>{{ item.label }}</strong>
              <em>{{ item.range }}</em>
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
  border: 1px solid rgba(15, 23, 42, 0.1);
  border-radius: 20px;
  background: #ffffff;
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
  background: #f8fbfa;
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

.confidence-legend {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  margin-top: 14px;
  padding: 12px 14px;
  border-radius: 14px;
  background: #f8fbfa;
  box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.06);
}

.confidence-legend > span {
  color: #64748b;
  font-size: 13px;
  font-weight: 800;
}

.confidence-legend__item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #0f172a;
  font-size: 13px;
}

.confidence-legend__item i {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.12);
}

.confidence-legend__item strong {
  font-weight: 800;
}

.confidence-legend__item em {
  color: #64748b;
  font-style: normal;
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
