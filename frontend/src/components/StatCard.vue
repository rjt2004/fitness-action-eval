<script setup>
defineProps({
  label: {
    type: String,
    default: "",
  },
  value: {
    type: [String, Number],
    default: "",
  },
  accent: {
    type: String,
    default: "#0f766e",
  },
});
</script>

<template>
  <div class="stat-card soft-card" :style="{ '--accent': accent }">
    <div class="stat-card__label">{{ label }}</div>
    <div class="stat-card__value">{{ value }}</div>
  </div>
</template>

<style scoped>
.stat-card {
  padding: 22px;
  border-top: 4px solid var(--accent);
}

.stat-card__label {
  color: #64748b;
  font-size: 14px;
}

.stat-card__value {
  margin-top: 12px;
  font-size: 36px;
  font-weight: 800;
}
</style>
