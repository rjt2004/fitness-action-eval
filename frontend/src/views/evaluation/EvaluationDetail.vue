<script setup>
import { computed, onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import {
  getEvaluationTaskDetail,
  getEvaluationTaskHints,
} from "@/api/evaluation";
import HintTable from "@/components/HintTable.vue";
import VideoPlayer from "@/components/VideoPlayer.vue";

const route = useRoute();
const loading = ref(false);
const detail = ref(null);
const hints = ref([]);

function toMediaUrl(path) {
  if (!path) {
    return "";
  }
  const normalized = path.replaceAll("\\", "/");
  const index = normalized.indexOf("/media/");
  if (index >= 0) {
    return normalized.slice(index);
  }
  return normalized.startsWith("media/") ? `/${normalized}` : normalized;
}

const phasePlots = computed(() =>
  (detail.value?.result_payload?.phase_plots || []).map((item) => ({
    ...item,
    imageUrl: toMediaUrl(item.image_path),
  })),
);

async function loadData() {
  loading.value = true;
  try {
    const id = route.params.id;
    const [taskDetail, hintData] = await Promise.all([
      getEvaluationTaskDetail(id),
      getEvaluationTaskHints(id),
    ]);
    detail.value = taskDetail;
    hints.value = hintData;
  } finally {
    loading.value = false;
  }
}

onMounted(loadData);
</script>

<template>
  <div class="page-shell">
    <h2 class="page-title">评估详情</h2>

    <el-skeleton :loading="loading" animated :rows="10">
      <template #default>
        <section class="soft-card detail-panel">
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
          </div>
        </section>

        <section class="soft-card detail-panel" style="margin-top: 20px">
          <h3 class="section-title">分阶段 DTW 曲线</h3>
          <div v-if="phasePlots.length" class="phase-plot-grid">
            <div
              v-for="item in phasePlots"
              :key="item.phase_id"
              class="phase-plot-card"
            >
              <div class="phase-plot-card__header">
                <div>
                  <div class="phase-plot-card__title">{{ item.phase_name }}</div>
                  <div class="phase-plot-card__cue">{{ item.cue }}</div>
                </div>
                <div class="phase-plot-card__meta">
                  <span>分数 {{ item.score }}</span>
                  <span>距离 {{ item.normalized_distance }}</span>
                </div>
              </div>
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
          <HintTable :items="hints" />
        </section>
      </template>
    </el-skeleton>
  </div>
</template>

<style scoped>
.detail-panel {
  padding: 22px;
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
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}

.phase-plot-card {
  padding: 16px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 18px;
  background: #ffffff;
}

.phase-plot-card__header {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  margin-bottom: 12px;
}

.phase-plot-card__title {
  font-size: 18px;
  font-weight: 700;
}

.phase-plot-card__cue {
  margin-top: 6px;
  color: #64748b;
  font-size: 13px;
  line-height: 1.6;
}

.phase-plot-card__meta {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 120px;
  color: #0f766e;
  font-size: 13px;
  font-weight: 700;
  text-align: right;
}

.phase-plot-card__image {
  width: 100%;
  border-radius: 12px;
  border: 1px solid rgba(15, 23, 42, 0.08);
}
</style>
