<script setup>
import { onMounted, ref } from "vue";
import { getLiveSessionDetail, getLiveSessions } from "@/api/liveSession";
import VideoPlayer from "@/components/VideoPlayer.vue";

const loading = ref(false);
const tableData = ref([]);
const drawerVisible = ref(false);
const currentDetail = ref(null);

function statusLabel(status) {
  if (status === "success") return "运行完成";
  if (status === "failed") return "运行失败";
  if (status === "stopped") return "已停止";
  if (status === "running") return "运行中";
  return "待启动";
}

async function loadData() {
  loading.value = true;
  try {
    tableData.value = await getLiveSessions();
  } finally {
    loading.value = false;
  }
}

async function handleView(row) {
  currentDetail.value = await getLiveSessionDetail(row.id);
  drawerVisible.value = true;
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
            {{ statusLabel(row.status) }}
          </template>
        </el-table-column>
        <el-table-column prop="avg_score" label="平均分" width="100" />
        <el-table-column prop="matched_frames" label="匹配帧数" width="120" />
        <el-table-column prop="created_at" label="创建时间" min-width="180" />
        <el-table-column label="操作" width="120">
          <template #default="{ row }">
            <el-button size="small" type="primary" @click="handleView(row)">
              查看
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </section>

    <el-drawer v-model="drawerVisible" title="会话详情" size="45%">
      <div v-if="currentDetail" class="live-detail">
        <el-descriptions :column="1" border>
          <el-descriptions-item label="会话名称">
            {{ currentDetail.session_name }}
          </el-descriptions-item>
          <el-descriptions-item label="状态">
            {{ statusLabel(currentDetail.status) }}
          </el-descriptions-item>
          <el-descriptions-item label="平均分">
            {{ currentDetail.avg_score }}
          </el-descriptions-item>
          <el-descriptions-item label="匹配帧数">
            {{ currentDetail.matched_frames }}
          </el-descriptions-item>
          <el-descriptions-item label="最终动作阶段">
            {{ currentDetail.final_phase_name || "--" }}
          </el-descriptions-item>
          <el-descriptions-item label="动作要领">
            {{ currentDetail.final_phase_cue || "--" }}
          </el-descriptions-item>
        </el-descriptions>

        <div style="margin-top: 18px">
          <VideoPlayer :path="currentDetail.output_video_path" />
        </div>
      </div>
    </el-drawer>
  </div>
</template>

<style scoped>
.page-panel {
  padding: 22px;
}

.live-detail {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
</style>
