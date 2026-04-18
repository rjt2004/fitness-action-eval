<script setup>
import { useRouter } from "vue-router";
import { useAuthStore } from "@/stores/auth";

defineProps({
  title: {
    type: String,
    default: "",
  },
  user: {
    type: Object,
    default: null,
  },
});

const router = useRouter();
const authStore = useAuthStore();

function handleLogout() {
  authStore.clearAuth();
  router.push({ name: "login" });
}
</script>

<template>
  <header class="app-header soft-card">
    <div>
      <div class="app-header__eyebrow">Baduanjin Fitness Action Eval</div>
      <h1 class="app-header__title">{{ title }}</h1>
    </div>
    <div class="app-header__meta">
      <el-tag effect="dark" type="success">{{ user?.role || "guest" }}</el-tag>
      <span class="app-header__name">{{ user?.real_name || user?.username || "未登录" }}</span>
      <el-button plain type="danger" @click="handleLogout">退出登录</el-button>
    </div>
  </header>
</template>

<style scoped>
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 18px 18px 0;
  padding: 18px 22px;
}

.app-header__eyebrow {
  margin-bottom: 4px;
  color: #0f766e;
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.app-header__title {
  margin: 0;
  font-size: 24px;
}

.app-header__meta {
  display: flex;
  align-items: center;
  gap: 12px;
}

.app-header__name {
  font-weight: 600;
}
</style>
