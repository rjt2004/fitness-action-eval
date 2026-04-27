<script setup>
import { onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import { ElMessage, ElMessageBox } from "element-plus";
import { deleteLiveSession, getLiveSessions } from "@/api/liveSession";

const router = useRouter();
const loading = ref(false);
const tableData = ref([]);

function statusLabel(status) {
  if (status === "success") return "运行完成";
  if (status === "failed") return "运行失败";
  if (status === "stopped") return "已停止";
  if (status === "paused") return "已暂停";
  if (status === "running") return "运行中";
  return "待启动";
}

function statusType(status) {
  if (status === "success") return "success";
  if (status === "failed") return "danger";
  if (status === "stopped") return "info";
  if (status === "paused") return "warning";
  if (status === "running") return "warning";
  return "info";
}

async function loadData() {
  loading.value = true;
  try {
    tableData.value = await getLiveSessions();
  } finally {
    loading.value = false;
  }
}

function handleView(row) {
  router.push({ name: "live-detail", params: { id: row.id } });
}

async function handleDelete(row) {
  await ElMessageBox.confirm(`确认删除实时会话“${row.session_name || row.session_no}”吗？`, "删除确认", {
    type: "warning",
  });
  await deleteLiveSession(row.id);
  ElMessage.success("实时会话已删除");
  await loadData();
}

onMounted(loadData);
</script>

<template>
  <div class="page-shell">
    <h2 class="page-title">会话记录</h2>

    <section class="soft-card page-panel">
      <el-table v-loading="loading" :data="tableData" stripe style="width: 100%">
        <el-table-column prop="session_no" label="会话编号" min-width="220" />
        <el-table-column prop="session_name" label="会话名称" min-width="180" />
        <el-table-column label="状态" width="120">
          <template #default="{ row }">
            <el-tag :type="statusType(row.status)" effect="light">
              {{ statusLabel(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="avg_score" label="最终分数" width="110" />
        <el-table-column prop="hint_count" label="提示数" width="100" />
        <el-table-column prop="created_at" label="创建时间" min-width="180" />
        <el-table-column label="操作" min-width="180">
          <template #default="{ row }">
            <div class="action-row">
              <el-button size="small" type="primary" @click="handleView(row)">查看详情</el-button>
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

.action-row {
  display: flex;
  gap: 8px;
}
</style>
