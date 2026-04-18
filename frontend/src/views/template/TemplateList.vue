<script setup>
import { onMounted, reactive, ref } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import {
  buildTemplate,
  deleteTemplate,
  getTemplateList,
  uploadTemplate,
} from "@/api/template";

const loading = ref(false);
const tableData = ref([]);
const uploadFile = ref(null);

const form = reactive({
  template_name: "八段锦标准模板",
  frame_stride: 4,
  smooth_window: 5,
});

function statusType(status) {
  if (status === "ready") return "success";
  if (status === "failed") return "danger";
  return "warning";
}

function statusLabel(status) {
  if (status === "ready") return "已生成";
  if (status === "failed") return "生成失败";
  return "草稿";
}

async function loadData() {
  tableData.value = await getTemplateList();
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
  payload.append("source_video", uploadFile.value);

  loading.value = true;
  try {
    const uploaded = await uploadTemplate(payload);
    const built = await buildTemplate(uploaded.id);
    ElMessage.success(`模板已生成：${built.template_name}`);
    uploadFile.value = null;
    await loadData();
  } finally {
    loading.value = false;
  }
}

async function handleRebuild(row) {
  loading.value = true;
  try {
    const data = await buildTemplate(row.id);
    ElMessage.success(`模板已重新生成：${data.template_name}`);
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

onMounted(loadData);
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
      </el-form>

      <div class="template-upload">
        <el-upload
          :auto-upload="false"
          :limit="1"
          accept="video/*"
          :on-change="handleFileChange"
        >
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
        <el-table-column prop="template_name" label="模板名称" min-width="240" />
        <el-table-column label="状态" width="120">
          <template #default="{ row }">
            <el-tag :type="statusType(row.status)">{{ statusLabel(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="frame_stride" label="抽帧" width="100" />
        <el-table-column prop="smooth_window" label="平滑" width="100" />
        <el-table-column prop="created_by_name" label="创建人" width="120" />
        <el-table-column prop="created_at" label="创建时间" min-width="180" />
        <el-table-column label="操作" width="220">
          <template #default="{ row }">
            <div class="row-actions">
              <el-button
                size="small"
                type="primary"
                :disabled="loading"
                @click="handleRebuild(row)"
              >
                重新生成
              </el-button>
              <el-button
                size="small"
                type="danger"
                plain
                :disabled="loading"
                @click="handleDelete(row)"
              >
                删除
              </el-button>
            </div>
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

.row-actions {
  display: flex;
  gap: 8px;
}
</style>
