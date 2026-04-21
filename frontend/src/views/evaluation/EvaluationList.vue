<script setup>
import { onBeforeUnmount, onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import { ElMessage, ElMessageBox } from "element-plus";
import { deleteEvaluationTask, getEvaluationTasks } from "@/api/evaluation";

const router = useRouter();
const tableData = ref([]);
let pollTimer = null;

function statusLabel(status) {
  if (status === "success") return "评估成功";
  if (status === "failed") return "评估失败";
  if (status === "running") return "评估中";
  return "待处理";
}

function statusType(status) {
  if (status === "success") return "success";
  if (status === "failed") return "danger";
  if (status === "running") return "warning";
  return "info";
}

function clearPolling() {
  if (pollTimer) {
    window.clearTimeout(pollTimer);
    pollTimer = null;
  }
}

function schedulePolling() {
  clearPolling();
  if (!tableData.value.some((item) => ["pending", "running"].includes(item.status))) return;
  pollTimer = window.setTimeout(loadData, 3000);
}

async function loadData() {
  tableData.value = await getEvaluationTasks();
  schedulePolling();
}

function goDetail(row) {
  router.push({ name: "evaluation-detail", params: { id: row.id } });
}

async function handleDelete(row) {
  await ElMessageBox.confirm(`确认删除评估任务“${row.task_name || row.task_no}”吗？`, "删除确认", {
    type: "warning",
  });
  await deleteEvaluationTask(row.id);
  ElMessage.success("评估任务已删除");
  await loadData();
}

onMounted(loadData);
onBeforeUnmount(clearPolling);
</script>

<template>
  <div class="page-shell">
    <h2 class="page-title">评估结果</h2>

    <section class="soft-card page-panel">
      <el-table :data="tableData" stripe style="width: 100%">
        <el-table-column prop="task_no" label="任务编号" min-width="220" />
        <el-table-column prop="task_name" label="任务名称" min-width="180" />
        <el-table-column label="状态" width="120">
          <template #default="{ row }">
            <el-tag :type="statusType(row.status)" effect="light">
              {{ statusLabel(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="进度" min-width="220">
          <template #default="{ row }">
            <div v-if="['pending', 'running'].includes(row.status)" class="progress-cell">
              <el-progress :percentage="Number(row.progress_percent || 0)" :stroke-width="12" />
              <span class="progress-cell__text">{{ row.progress_text || "处理中" }}</span>
            </div>
            <span v-else>{{ row.progress_text || "--" }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="score" label="得分" width="100" />
        <el-table-column prop="hint_count" label="提示数" width="100" />
        <el-table-column prop="created_at" label="创建时间" min-width="180" />
        <el-table-column label="操作" min-width="180">
          <template #default="{ row }">
            <div class="action-row">
              <el-button size="small" type="primary" @click="goDetail(row)">查看详情</el-button>
              <el-button size="small" type="danger" plain @click="handleDelete(row)">删除</el-button>
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

.progress-cell {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.progress-cell__text {
  color: #64748b;
  font-size: 12px;
}

.action-row {
  display: flex;
  gap: 8px;
}
</style>
