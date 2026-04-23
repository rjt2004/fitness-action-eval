<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { getEvaluationTaskDetail, getEvaluationTaskHints } from "@/api/evaluation";
import VideoPlayer from "@/components/VideoPlayer.vue";

const route = useRoute();
const loading = ref(false);
const detail = ref(null);
const hints = ref([]);
let pollTimer = null;

function toMediaUrl(path) {
  if (!path) return "";
  const normalized = path.replaceAll("\\", "/");
  const index = normalized.indexOf("/media/");
  if (index >= 0) return normalized.slice(index);
  return normalized.startsWith("media/") ? `/${normalized}` : normalized;
}

const phasePlots = computed(() =>
  (detail.value?.result_payload?.phase_plots || []).map((item) => ({
    ...item,
    imageUrl: toMediaUrl(item.image_path),
  })),
);

const isRunning = computed(() => ["pending", "running"].includes(detail.value?.status || ""));

function clearPolling() {
  if (pollTimer) {
    window.clearTimeout(pollTimer);
    pollTimer = null;
  }
}

function schedulePolling() {
  clearPolling();
  if (!isRunning.value) return;
  pollTimer = window.setTimeout(() => {
    loadData({ silent: true });
  }, 3000);
}

async function loadData({ silent = false } = {}) {
  if (!silent) loading.value = true;
  try {
    const id = route.params.id;
    const taskDetail = await getEvaluationTaskDetail(id);
    detail.value = taskDetail;

    if (taskDetail.status === "success") {
      hints.value = await getEvaluationTaskHints(id);
    } else if (taskDetail.status === "failed") {
      hints.value = [];
    }
  } finally {
    if (!silent) loading.value = false;
    schedulePolling();
  }
}

onMounted(() => loadData());
onBeforeUnmount(clearPolling);
</script>

<template>
  <div class="page-shell">
    <h2 class="page-title">评估详情</h2>

    <el-skeleton :loading="loading" animated :rows="10">
      <template #default>
        <section v-if="detail && isRunning" class="soft-card detail-panel progress-panel">
          <div class="progress-panel__header">
            <div>
              <h3 class="section-title">任务进度</h3>
              <p class="progress-panel__text">
                {{ detail.progress_text || "任务正在处理中" }}
              </p>
            </div>
            <el-tag type="warning">{{ detail.status === "pending" ? "排队中" : "处理中" }}</el-tag>
          </div>
          <el-progress :percentage="Number(detail.progress_percent || 0)" :stroke-width="18" status="success" />
        </section>

        <section v-if="detail?.error_message" class="soft-card detail-panel" style="margin-top: 20px">
          <h3 class="section-title">错误信息</h3>
          <el-alert :title="detail.error_message" type="error" :closable="false" show-icon />
        </section>

        <section class="soft-card detail-panel" style="margin-top: 20px">
          <h3 class="section-title">参考动作与待测动作对比视频</h3>
          <VideoPlayer :path="detail?.result_video_path" />
        </section>

        <section class="soft-card detail-panel" style="margin-top: 20px">
          <h3 class="section-title">任务信息</h3>
          <div class="task-info-grid">
            <div class="task-info-item">
              <div class="task-info-item__label">任务编号</div>
              <div class="task-info-item__value">{{ detail?.task_no || "--" }}</div>
            </div>
            <div class="task-info-item">
              <div class="task-info-item__label">任务名称</div>
              <div class="task-info-item__value">{{ detail?.task_name || "--" }}</div>
            </div>
            <div class="task-info-item">
              <div class="task-info-item__label">评估用户</div>
              <div class="task-info-item__value">{{ detail?.username || "--" }}</div>
            </div>
            <div class="task-info-item">
              <div class="task-info-item__label">使用模板</div>
              <div class="task-info-item__value">{{ detail?.template?.template_name || "--" }}</div>
            </div>
            <div class="task-info-item">
              <div class="task-info-item__label">归一化距离</div>
              <div class="task-info-item__value">{{ detail?.normalized_distance ?? "--" }}</div>
            </div>
            <div class="task-info-item">
              <div class="task-info-item__label">抽帧步长</div>
              <div class="task-info-item__value">{{ detail?.frame_stride ?? "--" }}</div>
            </div>
            <div class="task-info-item">
              <div class="task-info-item__label">平滑窗口</div>
              <div class="task-info-item__value">{{ detail?.smooth_window ?? "--" }}</div>
            </div>
            <div class="task-info-item">
              <div class="task-info-item__label">姿态模型</div>
              <div class="task-info-item__value">{{ detail?.pose_model_label || detail?.pose_model || "--" }}</div>
            </div>
            <div class="task-info-item">
              <div class="task-info-item__label">提示阈值</div>
              <div class="task-info-item__value">{{ detail?.hint_threshold ?? "--" }}</div>
            </div>
            <div class="task-info-item">
              <div class="task-info-item__label">提示间隔</div>
              <div class="task-info-item__value">{{ detail?.hint_min_interval ?? "--" }}</div>
            </div>
            <div class="task-info-item">
              <div class="task-info-item__label">提示上限</div>
              <div class="task-info-item__value">{{ detail?.max_hints ?? "--" }}</div>
            </div>
          </div>
        </section>

        <section class="soft-card detail-panel" style="margin-top: 20px">
          <h3 class="section-title">分阶段 DTW 曲线</h3>
          <div v-if="phasePlots.length" class="phase-plot-grid">
            <div v-for="item in phasePlots" :key="item.phase_id" class="phase-plot-card">
              <div class="phase-plot-card__title">{{ item.phase_name }}</div>
              <img
                v-if="item.imageUrl"
                :src="item.imageUrl"
                :alt="`${item.phase_name} DTW 图`"
                class="phase-plot-card__image"
              />
            </div>
          </div>
          <el-empty v-else description="当前任务还没有分阶段 DTW 图" />
        </section>

        <section class="soft-card detail-panel" style="margin-top: 20px">
          <h3 class="section-title">纠错提示</h3>
          <div v-if="hints.length" class="hint-scroll">
            <div
              v-for="(item, index) in hints"
              :key="`${item.query_time_s}-${index}`"
              class="hint-item"
            >
              <div class="hint-item__meta">
                <span>{{ item.query_time_s }}s</span>
                <span>{{ item.phase_name || "--" }}</span>
                <span>{{ item.part || "--" }}</span>
              </div>
              <div class="hint-item__message">{{ item.message }}</div>
            </div>
          </div>
          <el-empty v-else description="当前任务暂无提示记录" />
        </section>
      </template>
    </el-skeleton>
  </div>
</template>

<style scoped>
.detail-panel {
  padding: 22px;
}

.progress-panel__header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  margin-bottom: 16px;
}

.progress-panel__text {
  margin: 8px 0 0;
  color: #64748b;
}

.task-info-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 14px;
}

.task-info-item {
  padding: 16px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 16px;
  background: #f8fbfa;
}

.task-info-item__label {
  color: #64748b;
  font-size: 13px;
}

.task-info-item__value {
  margin-top: 8px;
  font-weight: 700;
  word-break: break-all;
}

.phase-plot-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 14px;
}

.phase-plot-card {
  padding: 12px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 16px;
  background: #ffffff;
}

.phase-plot-card__title {
  margin-bottom: 10px;
  font-size: 14px;
  font-weight: 700;
  text-align: center;
  line-height: 1.5;
}

.phase-plot-card__image {
  display: block;
  width: 100%;
  border-radius: 10px;
  border: 1px solid rgba(15, 23, 42, 0.08);
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
  background: #ffffff;
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

@media (max-width: 1400px) {
  .phase-plot-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

@media (max-width: 1200px) {
  .task-info-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .phase-plot-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 768px) {
  .phase-plot-grid {
    grid-template-columns: 1fr;
  }
}
</style>
