<script setup>
defineProps({
  items: {
    type: Array,
    default: () => [],
  },
});
</script>

<template>
  <el-table :data="items" stripe style="width: 100%">
    <el-table-column prop="query_time_s" label="时间(s)" min-width="100" />
    <el-table-column prop="phase_name" label="动作阶段" min-width="180" />
    <el-table-column prop="part" label="身体部位" min-width="120" />
    <el-table-column prop="message" label="纠错提示" min-width="280" />
    <el-table-column prop="part_error" label="部位误差" min-width="110" />
  </el-table>
</template>
