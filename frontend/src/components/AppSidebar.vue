<script setup>
import { computed } from "vue";
import { useRoute, useRouter } from "vue-router";
import { useAuthStore } from "@/stores/auth";

const route = useRoute();
const router = useRouter();
const authStore = useAuthStore();

const items = computed(() => {
  if (authStore.user?.role === "admin") {
    return [
      { name: "dashboard", label: "系统概览", icon: "DataAnalysis" },
      { name: "templates", label: "模板管理", icon: "FolderOpened" },
      { name: "evaluation-list", label: "评估结果", icon: "Tickets" },
      { name: "live-list", label: "会话记录", icon: "Timer" },
    ];
  }

  return [
    { name: "evaluation-create", label: "离线评估", icon: "VideoPlay" },
    { name: "evaluation-list", label: "评估结果", icon: "Tickets" },
    { name: "live", label: "实时跟练", icon: "Monitor" },
    { name: "live-list", label: "会话记录", icon: "Timer" },
  ];
});

function handleSelect(name) {
  router.push({ name });
}
</script>

<template>
  <aside class="app-sidebar">
    <div class="app-sidebar__brand">
      <div class="app-sidebar__badge">FIT</div>
      <div>
        <div class="app-sidebar__title">健身动作评价</div>
      </div>
    </div>

    <el-menu
      :default-active="route.name"
      class="app-sidebar__menu"
      @select="handleSelect"
    >
      <el-menu-item
        v-for="item in items"
        :key="item.name"
        :index="item.name"
      >
        <el-icon><component :is="item.icon" /></el-icon>
        <span>{{ item.label }}</span>
      </el-menu-item>
    </el-menu>
  </aside>
</template>

<style scoped>
.app-sidebar {
  width: 250px;
  padding: 24px 16px;
  color: #ecfeff;
  background:
    linear-gradient(180deg, rgba(15, 118, 110, 0.96), rgba(17, 94, 89, 0.98)),
    linear-gradient(135deg, rgba(255, 255, 255, 0.12), transparent 35%);
}

.app-sidebar__brand {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 28px;
  padding: 8px 10px;
}

.app-sidebar__badge {
  display: grid;
  place-items: center;
  width: 44px;
  height: 44px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.16);
  font-weight: 800;
}

.app-sidebar__title {
  font-size: 18px;
  font-weight: 800;
}

.app-sidebar__menu {
  border: 0;
  background: transparent;
}

:deep(.el-menu-item) {
  margin-bottom: 8px;
  border-radius: 14px;
  color: #ecfeff;
}

:deep(.el-menu-item.is-active) {
  background: rgba(255, 255, 255, 0.18);
  color: #ffffff;
}

:deep(.el-menu-item:hover) {
  background: rgba(255, 255, 255, 0.12);
}
</style>
