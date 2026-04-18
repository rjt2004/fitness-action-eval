<script setup>
import { onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import { getEvaluationTasks } from "@/api/evaluation";

const router = useRouter();
const tableData = ref([]);

function statusLabel(status) {
  if (status === "success") return "评估成功";
  if (status === "failed") return "评估失败";
  if (status === "running") return "评估中";
  return "待处理";
}

async function loadData() {
  tableData.value = await getEvaluationTasks();
}

function goDetail(row) {
  router.push({ name: "evaluation-detail", params: { id: row.id } });
}

onMounted(loadData);
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
            {{ statusLabel(row.status) }}
          </template>
        </el-table-column>
        <el-table-column prop="score" label="得分" width="100" />
        <el-table-column prop="hint_count" label="提示数" width="100" />
        <el-table-column prop="created_at" label="创建时间" min-width="180" />
        <el-table-column label="操作" width="120">
          <template #default="{ row }">
            <el-button size="small" type="primary" @click="goDetail(row)">
              查看详情
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
</style>
