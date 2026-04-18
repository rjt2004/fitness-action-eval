<script setup>
function phaseTypeLabel(type) {
  if (type === "reference") return "参考阶段";
  if (type === "query") return "测试阶段";
  return "动作阶段";
}

defineProps({
  items: {
    type: Array,
    default: () => [],
  },
});
</script>

<template>
  <div class="phase-timeline">
    <div
      v-for="item in items"
      :key="item.id"
      class="phase-timeline__item soft-card"
    >
      <div class="phase-timeline__header">
        <div class="phase-timeline__name">{{ item.phase_name || "--" }}</div>
        <el-tag size="small" effect="plain">{{ phaseTypeLabel(item.phase_type) }}</el-tag>
      </div>
      <div class="phase-timeline__meta">
        <span>序号：{{ item.phase_id ?? "--" }}</span>
        <span>区间：{{ item.start_time_s ?? 0 }}s - {{ item.end_time_s ?? 0 }}s</span>
      </div>
      <div class="phase-timeline__cue">{{ item.cue || "暂无动作要领" }}</div>
    </div>
  </div>
</template>

<style scoped>
.phase-timeline {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.phase-timeline__item {
  padding: 16px 18px;
}

.phase-timeline__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.phase-timeline__name {
  font-size: 16px;
  font-weight: 700;
}

.phase-timeline__meta {
  display: flex;
  gap: 18px;
  margin-top: 10px;
  color: #64748b;
  font-size: 13px;
}

.phase-timeline__cue {
  margin-top: 10px;
  color: #334155;
}
</style>
