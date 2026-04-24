<script setup>
import { computed } from "vue";
import { useRoute } from "vue-router";
import { useAuthStore } from "@/stores/auth";
import AppHeader from "@/components/AppHeader.vue";
import AppSidebar from "@/components/AppSidebar.vue";

const route = useRoute();
const authStore = useAuthStore();

const pageTitle = computed(() => route.meta?.title || "健身动作评价");
</script>

<template>
  <div class="main-layout">
    <AppSidebar />
    <div class="main-layout__content">
      <AppHeader :title="pageTitle" :user="authStore.user" />
      <main class="main-layout__main">
        <router-view />
      </main>
    </div>
  </div>
</template>

<style scoped>
.main-layout {
  display: flex;
  min-height: 100vh;
}

.main-layout__content {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-width: 0;
}

.main-layout__main {
  flex: 1;
  padding: 18px;
}
</style>
