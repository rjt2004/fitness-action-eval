<script setup>
import { onBeforeUnmount, onMounted, reactive, ref } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import { deleteTemplate, getTemplateDetail, getTemplateList, uploadTemplate } from "@/api/template";
import { getPoseModelOptions } from "@/api/system";

const loading = ref(false);
const tableData = ref([]);
const uploadFile = ref(null);
const poseModelOptions = ref([]);
const ruleDialogVisible = ref(false);
const ruleLoading = ref(false);
const selectedRuleTemplate = ref(null);
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

async function handleViewRules(row) {
  ruleDialogVisible.value = true;
  ruleLoading.value = true;
  try {
    selectedRuleTemplate.value = await getTemplateDetail(row.id);
  } finally {
    ruleLoading.value = false;
  }
}

function formatPriority(parts) {
  return Array.isArray(parts) && parts.length ? parts.join("、") : "--";
}

function formatHintTemplates(templates) {
  if (!templates || typeof templates !== "object") return [];
  return Object.entries(templates).map(([part, text]) => ({ part, text }));
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
        <el-table-column label="操作" width="190">
          <template #default="{ row }">
            <el-button size="small" plain @click="handleViewRules(row)">
              评分规则
            </el-button>
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

    <el-dialog v-model="ruleDialogVisible" title="模板评分规则" width="920px">
      <div v-loading="ruleLoading">
        <div v-if="selectedRuleTemplate" class="rule-summary">
          <div><strong>模板：</strong>{{ selectedRuleTemplate.template_name }}</div>
          <div><strong>规则：</strong>{{ selectedRuleTemplate.rule_config?.action_key || "baduanjin" }}</div>
          <div><strong>标准时长：</strong>{{ selectedRuleTemplate.rule_config?.standard_duration_s || "--" }}s</div>
        </div>
        <el-collapse v-if="selectedRuleTemplate?.phase_guides?.length" class="rule-collapse">
          <el-collapse-item
            v-for="phase in selectedRuleTemplate.phase_guides"
            :key="phase.phase_id"
            :title="`${phase.phase_id}. ${phase.phase_name}`"
            :name="String(phase.phase_id)"
          >
            <div class="rule-block">
              <div><strong>动作要领：</strong>{{ phase.cue || "--" }}</div>
              <div><strong>重点部位：</strong>{{ formatPriority(phase.feedback_priority) }}</div>
            </div>
            <el-table :data="formatHintTemplates(phase.hint_templates)" size="small" border>
              <el-table-column prop="part" label="部位" width="140" />
              <el-table-column prop="text" label="提示文本" min-width="520" />
            </el-table>
          </el-collapse-item>
        </el-collapse>
        <el-empty v-else description="该模板暂无评分规则" />
      </div>
    </el-dialog>
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

.rule-summary {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 14px;
  color: #334155;
}

.rule-collapse {
  max-height: 620px;
  overflow-y: auto;
}

.rule-block {
  display: grid;
  gap: 8px;
  margin-bottom: 12px;
  color: #334155;
}
</style>
