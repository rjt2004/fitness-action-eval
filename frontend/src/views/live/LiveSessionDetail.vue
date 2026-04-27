<script setup>
import { computed, onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { getLiveSessionDetail } from "@/api/liveSession";

const route = useRoute();
const loading = ref(false);
const detail = ref(null);

function toMediaUrl(path) {
  if (!path) return "";
  const normalized = path.replaceAll("\\", "/");
  const index = normalized.indexOf("/media/");
  if (index >= 0) return normalized.slice(index);
  return normalized.startsWith("media/") ? `/${normalized}` : normalized;
}

const hintItems = computed(() => detail.value?.summary_payload?.hints || []);
const errorFrameItems = computed(() => detail.value?.summary_payload?.error_frames || []);
const finalScore = computed(() => detail.value?.summary_payload?.score_0_100 ?? detail.value?.avg_score ?? "");

async function loadData() {
  loading.value = true;
  try {
    detail.value = await getLiveSessionDetail(route.params.id);
  } finally {
    loading.value = false;
  }
}

onMounted(loadData);
</script>

<template>
  <div class="page-shell">
    <h2 class="page-title">会话详情</h2>

    <el-skeleton :loading="loading" animated :rows="10">
      <template #default>
        <section class="soft-card detail-panel">
          <h3 class="section-title">错误动作帧</h3>
          <div v-if="errorFrameItems.length" class="error-frame-grid">
            <article
              v-for="(item, index) in errorFrameItems"
              :key="`${item.query_time_s}-${index}`"
              class="error-frame-card"
            >
              <el-image
                class="error-frame-card__image"
                :src="toMediaUrl(item.image_path)"
                :preview-src-list="errorFrameItems.map((frame) => toMediaUrl(frame.image_path))"
                :initial-index="index"
                fit="contain"
                preview-teleported
              />
              <div class="error-frame-card__body">
                <div class="error-frame-card__meta">
                  <span>{{ Number(item.query_time_s || 0).toFixed(1) }}s</span>
                  <span>{{ item.phase_name || "--" }}</span>
                  <span>{{ item.part || "--" }}</span>
                  <span>偏差 {{ item.local_error !== undefined ? Number(item.local_error).toFixed(3) : "--" }}</span>
                </div>
                <div class="error-frame-card__message">{{ item.message }}</div>
              </div>
            </article>
          </div>
          <el-empty v-else description="本次会话暂无保存的错误动作帧" />
        </section>

        <section class="soft-card detail-panel" style="margin-top: 20px">
          <h3 class="section-title">任务信息</h3>
          <div class="task-info-grid">
            <div class="task-info-item">
              <div class="task-info-item__label">会话编号</div>
              <div class="task-info-item__value">{{ detail?.session_no || "--" }}</div>
            </div>
            <div class="task-info-item">
              <div class="task-info-item__label">会话名称</div>
              <div class="task-info-item__value">{{ detail?.session_name || "--" }}</div>
            </div>
            <div class="task-info-item">
              <div class="task-info-item__label">用户</div>
              <div class="task-info-item__value">{{ detail?.username || "--" }}</div>
            </div>
            <div class="task-info-item">
              <div class="task-info-item__label">使用模板</div>
              <div class="task-info-item__value">{{ detail?.template?.template_name || "--" }}</div>
            </div>
            <div class="task-info-item">
              <div class="task-info-item__label">最终分数</div>
              <div class="task-info-item__value">{{ finalScore !== "" ? Number(finalScore).toFixed(2) : "--" }}</div>
            </div>
            <div class="task-info-item">
              <div class="task-info-item__label">归一化距离</div>
              <div class="task-info-item__value">{{ detail?.summary_payload?.normalized_dtw_distance ?? "--" }}</div>
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
          <h3 class="section-title">纠错提示</h3>
          <div v-if="hintItems.length" class="hint-scroll">
            <div
              v-for="(item, index) in hintItems"
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
          <el-empty v-else description="该会话暂无提示记录" />
        </section>
      </template>
    </el-skeleton>
  </div>
</template>

<style scoped>
.detail-panel {
  padding: 22px;
}

.error-frame-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}

.error-frame-card {
  overflow: hidden;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 18px;
  background: #ffffff;
}

.error-frame-card__image {
  display: block;
  width: 100%;
  min-height: 220px;
  background: #050505;
}

.error-frame-card__body {
  padding: 14px 16px 16px;
}

.error-frame-card__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  color: #64748b;
  font-size: 13px;
}

.error-frame-card__message {
  margin-top: 8px;
  color: #0f172a;
  font-weight: 700;
  line-height: 1.7;
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
  line-height: 1.7;
}

.hint-scroll {
  display: flex;
  flex-direction: column;
  gap: 12px;
  max-height: 420px;
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

@media (max-width: 1200px) {
  .error-frame-grid,
  .task-info-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
