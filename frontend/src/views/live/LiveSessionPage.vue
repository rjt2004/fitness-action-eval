<script setup>
import { onBeforeUnmount, onMounted, reactive, ref } from "vue";
import { ElMessage } from "element-plus";
import { getTemplateList } from "@/api/template";
import {
  getLiveSessionDetail,
  startLiveSession,
  stopLiveSession,
} from "@/api/liveSession";
import VideoPlayer from "@/components/VideoPlayer.vue";

const templates = ref([]);
const loading = ref(false);
const currentSession = ref(null);
let timer = null;

const form = reactive({
  template_id: "",
  session_name: "八段锦实时跟练",
  camera_source: "0",
  camera_width: 1280,
  camera_height: 720,
  camera_mirror: true,
  preview: false,
  frame_stride: 2,
  smooth_window: 5,
  score_scale: 100,
  hint_threshold: 0.18,
  ref_search_window: 20,
  max_frames: 300,
});

function statusLabel(status) {
  if (status === "success") return "运行完成";
  if (status === "failed") return "运行失败";
  if (status === "stopped") return "已停止";
  if (status === "running") return "运行中";
  return "待启动";
}

async function loadTemplates() {
  const data = await getTemplateList();
  templates.value = data.filter((item) => item.status === "ready");
  if (!form.template_id && templates.value.length) {
    form.template_id = templates.value[0].id;
  }
}

function stopPolling() {
  if (timer) {
    window.clearInterval(timer);
    timer = null;
  }
}

function startPolling(sessionId) {
  stopPolling();
  timer = window.setInterval(async () => {
    const data = await getLiveSessionDetail(sessionId);
    currentSession.value = data;
    if (["success", "failed", "stopped"].includes(data.status)) {
      stopPolling();
    }
  }, 2000);
}

async function handleStart() {
  loading.value = true;
  try {
    const payload = {
      ...form,
      camera_source:
        String(Number(form.camera_source)) === form.camera_source
          ? Number(form.camera_source)
          : form.camera_source,
    };
    const data = await startLiveSession(payload);
    currentSession.value = data;
    ElMessage.success("实时会话已启动");
    startPolling(data.id);
  } finally {
    loading.value = false;
  }
}

async function handleStop() {
  if (!currentSession.value?.id) {
    return;
  }
  await stopLiveSession(currentSession.value.id);
  ElMessage.success("已发送停止信号");
}

onMounted(loadTemplates);
onBeforeUnmount(stopPolling);
</script>

<template>
  <div class="page-shell">
    <h2 class="page-title">实时跟练</h2>

    <section class="soft-card page-panel">
      <h3 class="section-title">会话参数</h3>
      <el-form label-width="130px">
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
          </div>
          <div>
            <el-form-item label="镜像显示">
              <el-switch v-model="form.camera_mirror" />
            </el-form-item>
            <el-form-item label="本地预览">
              <el-switch v-model="form.preview" />
            </el-form-item>
            <el-form-item label="抽帧步长">
              <el-input-number v-model="form.frame_stride" :min="1" :max="10" />
            </el-form-item>
            <el-form-item label="平滑窗口">
              <el-input-number v-model="form.smooth_window" :min="1" :max="15" />
            </el-form-item>
            <el-form-item label="最大帧数">
              <el-input-number v-model="form.max_frames" :min="30" :max="5000" />
            </el-form-item>
          </div>
        </div>
        <el-form-item>
          <el-button type="primary" :loading="loading" @click="handleStart">
            启动会话
          </el-button>
          <el-button type="danger" plain @click="handleStop">停止会话</el-button>
        </el-form-item>
      </el-form>
    </section>

    <section class="grid-two" style="margin-top: 20px">
      <div class="soft-card page-panel">
        <h3 class="section-title">当前会话状态</h3>
        <el-descriptions :column="1" border>
          <el-descriptions-item label="状态">
            {{ statusLabel(currentSession?.status) }}
          </el-descriptions-item>
          <el-descriptions-item label="平均分">
            {{ currentSession?.avg_score ?? "--" }}
          </el-descriptions-item>
          <el-descriptions-item label="匹配帧数">
            {{ currentSession?.matched_frames ?? "--" }}
          </el-descriptions-item>
          <el-descriptions-item label="最终动作阶段">
            {{ currentSession?.final_phase_name || "--" }}
          </el-descriptions-item>
          <el-descriptions-item label="动作要领">
            {{ currentSession?.final_phase_cue || "--" }}
          </el-descriptions-item>
        </el-descriptions>
      </div>
      <VideoPlayer :path="currentSession?.output_video_path" />
    </section>
  </div>
</template>

<style scoped>
.page-panel {
  padding: 22px;
}
</style>
