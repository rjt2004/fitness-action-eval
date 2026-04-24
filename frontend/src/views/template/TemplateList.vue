<script setup>
import { onBeforeUnmount, onMounted, reactive, ref } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import { deleteTemplate, getTemplateList, uploadTemplate } from "@/api/template";
import { getPoseModelOptions } from "@/api/system";

const loading = ref(false);
const tableData = ref([]);
const uploadFile = ref(null);
const poseModelOptions = ref([]);
let pollTimer = null;

// 模板生成参数默认更偏向提取精度，而不是实时性。
const form = reactive({
  template_name: "八段锦标准模板",
  frame_stride: 4,
  smooth_window: 5,
  pose_model: "heavy",
});

function statusType(status) {
  if (status === "ready") return "success";
  if (status === "failed") return "danger";
  if (status === "building") return "warning";
  return "warning";
}

function statusLabel(status) {
  if (status === "ready") return "已生成";
  if (status === "failed") return "生成失败";
  if (status === "building") return "生成中";
  return "草稿";
}

function progressText(row) {
  if (row.progress_text) return row.progress_text;
  if (row.status === "ready") return "模板生成完成";
  if (row.status === "failed") return "模板生成失败";
  return "处理中";
}

function clearPolling() {
  if (pollTimer) {
    window.clearTimeout(pollTimer);
    pollTimer = null;
  }
}

function schedulePolling() {
  clearPolling();
  if (!tableData.value.some((item) => ["draft", "building"].includes(item.status))) return;
  pollTimer = window.setTimeout(loadData, 3000);
}

async function loadData() {
  tableData.value = await getTemplateList();
  schedulePolling();
}

async function loadPoseModelOptions() {
  poseModelOptions.value = await getPoseModelOptions();
}

function handleFileChange(file) {
  uploadFile.value = file.raw;
}

async function handleUploadAndBuild() {
  if (!uploadFile.value) {
    ElMessage.warning("请先选择标准视频");
    return;
  }

  const payload = new FormData();
  payload.append("template_name", form.template_name);
  payload.append("frame_stride", form.frame_stride);
  payload.append("smooth_window", form.smooth_window);
  payload.append("pose_model", form.pose_model);
  payload.append("source_video", uploadFile.value);

  loading.value = true;
  try {
    const uploaded = await uploadTemplate(payload);
    ElMessage.success(`模板已上传，正在生成：${uploaded.template_name}`);
    uploadFile.value = null;
    await loadData();
  } finally {
    loading.value = false;
  }
}

async function handleDelete(row) {
  await ElMessageBox.confirm(
    `确定删除模板“${row.template_name}”吗？删除后源视频和生成文件也会一起清理。`,
    "删除模板",
    {
      type: "warning",
      confirmButtonText: "删除",
      cancelButtonText: "取消",
    },
  );

  loading.value = true;
  try {
    await deleteTemplate(row.id);
    ElMessage.success("模板删除成功");
    await loadData();
  } finally {
    loading.value = false;
  }
}

onMounted(() => {
  loadPoseModelOptions();
  loadData();
});
onBeforeUnmount(clearPolling);
</script>

<template>
  <div class="page-shell">
    <h2 class="page-title">模板管理</h2>

    <section class="soft-card page-panel">
      <h3 class="section-title">上传并生成标准模板</h3>
      <el-form inline>
        <el-form-item label="模板名称">
          <el-input v-model="form.template_name" style="width: 220px" />
        </el-form-item>
        <el-form-item label="抽帧步长">
          <el-input-number v-model="form.frame_stride" :min="1" :max="10" />
        </el-form-item>
        <el-form-item label="平滑窗口">
          <el-input-number v-model="form.smooth_window" :min="1" :max="15" />
        </el-form-item>
        <el-form-item label="姿态模型">
          <el-select v-model="form.pose_model" style="width: 190px">
            <el-option
              v-for="item in poseModelOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>
      </el-form>

      <div class="template-upload">
        <el-upload :auto-upload="false" :limit="1" accept="video/*" :on-change="handleFileChange">
          <el-button type="primary" plain>选择标准视频</el-button>
        </el-upload>
        <el-button type="primary" :loading="loading" @click="handleUploadAndBuild">
          上传并生成模板
        </el-button>
      </div>
    </section>

    <section class="soft-card page-panel" style="margin-top: 20px">
      <h3 class="section-title">模板列表</h3>
      <el-table :data="tableData" stripe style="width: 100%">
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="template_name" label="模板名称" min-width="220" />
        <el-table-column label="状态" width="120">
          <template #default="{ row }">
            <el-tag :type="statusType(row.status)">{{ statusLabel(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="进度" min-width="220">
          <template #default="{ row }">
            <div v-if="['draft', 'building'].includes(row.status)" class="progress-cell">
              <el-progress :percentage="Number(row.progress_percent || 0)" :stroke-width="12" />
              <span class="progress-cell__text">{{ progressText(row) }}</span>
            </div>
            <span v-else>{{ progressText(row) }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="pose_model_label" label="姿态模型" width="170" />
        <el-table-column prop="frame_stride" label="抽帧" width="90" />
        <el-table-column prop="smooth_window" label="平滑" width="90" />
        <el-table-column prop="created_by_name" label="创建人" width="120" />
        <el-table-column prop="created_at" label="创建时间" min-width="180" />
        <el-table-column label="操作" width="120">
          <template #default="{ row }">
            <el-button
              size="small"
              type="danger"
              plain
              :disabled="loading || row.status === 'building'"
              @click="handleDelete(row)"
            >
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </section>
  </div>
</template>

<style scoped>
.page-panel {
  padding: 22px;
}

.template-upload {
  display: flex;
  align-items: center;
  gap: 12px;
}

.progress-cell {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.progress-cell__text {
  color: var(--text-muted);
  font-size: 12px;
}
</style>
