<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import { onBeforeRouteLeave } from "vue-router";
import { ElMessage } from "element-plus";
import { getTemplateDetail, getTemplateList } from "@/api/template";
import {
  getLiveSessionDetail,
  getLiveSessionPreviewFrame,
  startLiveSession,
  stopLiveSession,
} from "@/api/liveSession";

const templates = ref([]);
const selectedTemplate = ref(null);
const templateLoading = ref(false);
const loading = ref(false);
const currentSession = ref(null);
const referenceVideoRef = ref(null);
const previewFrameUrl = ref("");
const referencePaused = ref(false);
let timer = null;
let previewCounter = 0;
let leaveStopping = false;

const form = reactive({
  template_id: "",
  session_name: "实时跟练会话",
  camera_source: "0",
  camera_width: 640,
  camera_height: 360,
  camera_mirror: true,
  export_video: false,
  frame_stride: 4,
  smooth_window: 5,
  hint_threshold: 0.18,
  hint_min_interval: 8,
  max_hints: 40,
  ref_search_window: 20,
});

const sessionActive = computed(() =>
  ["pending", "running"].includes(currentSession.value?.status || ""),
);
const canPauseReference = computed(() => Boolean(currentSession.value?.id) && sessionActive.value && !referencePaused.value);
const canResumeReference = computed(() => Boolean(currentSession.value?.id) && sessionActive.value && referencePaused.value);
const liveHints = computed(() => currentSession.value?.summary_payload?.hints || []);

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
  referencePaused.value = false;
  el.play().catch(() => {});
}

function pauseReferenceVideo() {
  const el = referenceVideoRef.value;
  if (!el) return;
  referencePaused.value = true;
  el.pause();
}

function getStoredAccessToken() {
  try {
    const raw = window.localStorage.getItem("fitness_action_eval_auth");
    if (!raw) return "";
    const parsed = JSON.parse(raw);
    return parsed?.accessToken || "";
  } catch {
    return "";
  }
}

function sendStopKeepalive(sessionId) {
  const token = getStoredAccessToken();
  if (!token) return;
  fetch(`/api/live-session/sessions/${sessionId}/stop/`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
    },
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

async function loadTemplateDetail(templateId) {
  if (!templateId) {
    selectedTemplate.value = null;
    return;
  }
  templateLoading.value = true;
  try {
    selectedTemplate.value = await getTemplateDetail(templateId);
    if (selectedTemplate.value) {
      form.frame_stride = selectedTemplate.value.frame_stride || 4;
      form.smooth_window = selectedTemplate.value.smooth_window || 5;
    }
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
    // Ignore transient preview errors while the session is starting.
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
      camera_source:
        String(Number(form.camera_source)) === form.camera_source
          ? Number(form.camera_source)
          : form.camera_source,
      camera_width: form.camera_width,
      camera_height: form.camera_height,
      camera_mirror: form.camera_mirror,
      preview: false,
      export_video: form.export_video,
      frame_stride: form.frame_stride,
      smooth_window: form.smooth_window,
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

function handlePauseReference() {
  pauseReferenceVideo();
}

function handleResumeReference() {
  playReferenceVideo();
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

onMounted(loadTemplates);
onMounted(() => {
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
      <div class="left-column">
        <section class="soft-card page-panel">
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

            <div v-if="selectedTemplate" class="template-meta">
              <div><strong>模板名称：</strong>{{ selectedTemplate.template_name }}</div>
            </div>
          </div>
        </section>

        <section class="soft-card page-panel" style="margin-top: 20px">
          <h3 class="section-title">实时画面</h3>
          <img
            v-if="previewFrameUrl"
            :src="previewFrameUrl"
            class="preview-video live-video"
            alt="实时预览"
          />
          <el-empty v-else description="开始跟练后，这里会显示带提示的实时画面" />
        </section>
      </div>

      <div class="right-column">
        <section class="soft-card page-panel">
          <h3 class="section-title">会话参数</h3>
          <el-form label-width="120px">
            <div class="grid-two">
              <div>
                <el-form-item label="会话名称">
                  <el-input v-model="form.session_name" />
                </el-form-item>
                <el-form-item label="动作模板">
                  <el-select v-model="form.template_id" style="width: 100%">
                    <el-option
                      v-for="item in templates"
                      :key="item.id"
                      :label="item.template_name"
                      :value="item.id"
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
                <el-form-item label="导出视频">
                  <el-switch v-model="form.export_video" />
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
              <el-button :disabled="!canPauseReference" @click="handlePauseReference">暂停参考</el-button>
              <el-button type="success" plain :disabled="!canResumeReference" @click="handleResumeReference">
                继续参考
              </el-button>
              <el-button type="danger" plain @click="handleStop">停止</el-button>
            </el-form-item>
          </el-form>
        </section>

        <section class="soft-card page-panel" style="margin-top: 20px">
          <h3 class="section-title">提示列表</h3>
          <el-empty v-if="!liveHints.length" description="会话过程中产生的提示会显示在这里" />
          <div v-else class="hint-scroll">
            <div v-for="(item, index) in liveHints" :key="`${item.query_time_s}-${index}`" class="hint-item">
              <div class="hint-item__meta">
                <span>{{ item.query_time_s }}s</span>
                <span>{{ item.phase_name || "--" }}</span>
                <span>{{ item.part || "--" }}</span>
              </div>
              <div class="hint-item__message">{{ item.message }}</div>
            </div>
          </div>
        </section>
      </div>
    </div>
  </div>
</template>

<style scoped>
.content-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.left-column,
.right-column {
  min-width: 0;
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

.template-meta {
  display: grid;
  gap: 8px;
  margin-top: 14px;
  color: #334155;
}

.grid-two {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px 18px;
}

.hint-scroll {
  display: flex;
  flex-direction: column;
  gap: 12px;
  max-height: 360px;
  overflow-y: auto;
  padding-right: 4px;
}

.hint-item {
  padding: 14px 16px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 14px;
  background: #f8fbfa;
}

.hint-item__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  color: #64748b;
  font-size: 13px;
}

.hint-item__message {
  margin-top: 8px;
  color: #0f172a;
  line-height: 1.7;
}

@media (max-width: 1200px) {
  .content-grid,
  .grid-two {
    grid-template-columns: 1fr;
  }
}
</style>
