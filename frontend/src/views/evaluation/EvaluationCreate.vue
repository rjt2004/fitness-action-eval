<script setup>
import { onMounted, reactive, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { ElMessage } from "element-plus";
import { createEvaluationTask } from "@/api/evaluation";
import { getPoseModelOptions } from "@/api/system";
import { getTemplateDetail, getTemplateList } from "@/api/template";

const router = useRouter();
const loading = ref(false);
const templateLoading = ref(false);
const templates = ref([]);
const selectedTemplate = ref(null);
const poseModelOptions = ref([]);
const queryVideo = ref(null);

const form = reactive({
  template_id: "",
  task_name: "离线评估任务",
  frame_stride: 6,
  smooth_window: 3,
  pose_model: "follow_template",
  hint_threshold: 0.2,
  hint_min_interval: 12,
  max_hints: 24,
  export_video: false,
});

const runtimeModelOptions = [
  { value: "follow_template", label: "跟随模板模型" },
];

function toMediaUrl(path) {
  if (!path) return "";
  const normalized = path.replaceAll("\\", "/");
  const index = normalized.indexOf("/media/");
  if (index >= 0) return normalized.slice(index);
  return normalized.startsWith("media/") ? `/${normalized}` : normalized;
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
    if (selectedTemplate.value) {
      form.frame_stride = Math.max(selectedTemplate.value.frame_stride || 4, 6);
      form.smooth_window = Math.min(selectedTemplate.value.smooth_window || 5, 3);
      form.pose_model = "follow_template";
    }
  } finally {
    templateLoading.value = false;
  }
}

function handleFileChange(file) {
  queryVideo.value = file.raw;
}

async function handleSubmit() {
  if (!queryVideo.value) {
    ElMessage.warning("请先上传待测视频");
    return;
  }

  const payload = new FormData();
  payload.append("template_id", form.template_id);
  payload.append("task_name", form.task_name);
  payload.append("frame_stride", String(form.frame_stride));
  payload.append("smooth_window", String(form.smooth_window));
  payload.append("pose_model", form.pose_model);
  payload.append("hint_threshold", String(form.hint_threshold));
  payload.append("hint_min_interval", String(form.hint_min_interval));
  payload.append("max_hints", String(form.max_hints));
  payload.append("export_video", String(form.export_video));
  payload.append("query_video", queryVideo.value);

  loading.value = true;
  try {
    const result = await createEvaluationTask(payload);
    ElMessage.success("评估任务创建成功");
    router.push({ name: "evaluation-detail", params: { id: result.id } });
  } finally {
    loading.value = false;
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
});
</script>

<template>
  <div class="page-shell">
    <h2 class="page-title">离线评估</h2>

    <div class="content-grid">
      <section class="soft-card page-panel">
        <h3 class="section-title">参考视频</h3>
        <div v-loading="templateLoading">
          <video
            v-if="selectedTemplate?.source_video_path"
            class="reference-video"
            :src="toMediaUrl(selectedTemplate.source_video_path)"
            controls
            preload="metadata"
          />
          <el-empty v-else description="请选择可用模板" />

          <div v-if="selectedTemplate" class="template-meta">
            <div><strong>模板名称：</strong>{{ selectedTemplate.template_name }}</div>
            <div><strong>模板姿态模型：</strong>{{ selectedTemplate.pose_model_label || selectedTemplate.pose_model }}</div>
            <div><strong>模板抽帧：</strong>{{ selectedTemplate.frame_stride }}</div>
            <div><strong>模板平滑：</strong>{{ selectedTemplate.smooth_window }}</div>
          </div>
        </div>
      </section>

      <section class="soft-card page-panel">
        <h3 class="section-title">创建评估任务</h3>
        <el-form label-width="120px">
          <el-form-item label="任务名称">
            <el-input v-model="form.task_name" />
          </el-form-item>
          <el-form-item label="选择模板">
            <el-select v-model="form.template_id" style="width: 100%">
              <el-option
                v-for="item in templates"
                :key="item.id"
                :label="item.template_name"
                :value="item.id"
              />
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
          <el-form-item label="导出对比视频">
            <el-switch v-model="form.export_video" />
          </el-form-item>
          <el-form-item label="待测视频">
            <el-upload :auto-upload="false" :limit="1" :on-change="handleFileChange">
              <el-button type="primary" plain>选择视频文件</el-button>
            </el-upload>
          </el-form-item>
          <el-form-item>
            <el-button type="primary" :loading="loading" @click="handleSubmit">提交评估</el-button>
          </el-form-item>
        </el-form>
      </section>
    </div>
  </div>
</template>

<style scoped>
.content-grid {
  display: grid;
  grid-template-columns: 1.1fr 1fr;
  gap: 20px;
}

.page-panel {
  padding: 22px;
}

.reference-video {
  display: block;
  width: 100%;
  max-height: 360px;
  border-radius: 16px;
  background: #000;
}

.template-meta {
  display: grid;
  gap: 8px;
  margin-top: 16px;
  color: #334155;
}

@media (max-width: 1100px) {
  .content-grid {
    grid-template-columns: 1fr;
  }
}
</style>
