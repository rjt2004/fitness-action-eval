import { createRouter, createWebHistory } from "vue-router";

import MainLayout from "@/layout/MainLayout.vue";
import Login from "@/views/Login.vue";
import Dashboard from "@/views/Dashboard.vue";
import TemplateList from "@/views/template/TemplateList.vue";
import EvaluationCreate from "@/views/evaluation/EvaluationCreate.vue";
import EvaluationList from "@/views/evaluation/EvaluationList.vue";
import EvaluationDetail from "@/views/evaluation/EvaluationDetail.vue";
import LiveSessionPage from "@/views/live/LiveSessionPage.vue";
import LiveSessionList from "@/views/live/LiveSessionList.vue";
import LiveSessionDetail from "@/views/live/LiveSessionDetail.vue";
import { useAuthStore } from "@/stores/auth";

function defaultRouteByRole(role) {
  return role === "admin" ? { name: "dashboard" } : { name: "evaluation-create" };
}

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: "/login",
      name: "login",
      component: Login,
      meta: { public: true, title: "用户登录" },
    },
    {
      path: "/",
      component: MainLayout,
      children: [
        {
          path: "",
          name: "dashboard",
          component: Dashboard,
          meta: { role: "admin", title: "系统概览" },
        },
        {
          path: "templates",
          name: "templates",
          component: TemplateList,
          meta: { role: "admin", title: "模板管理" },
        },
        {
          path: "evaluation/create",
          name: "evaluation-create",
          component: EvaluationCreate,
          meta: { role: "user", title: "离线评估" },
        },
        {
          path: "evaluation/list",
          name: "evaluation-list",
          component: EvaluationList,
          meta: { title: "评估结果" },
        },
        {
          path: "evaluation/:id",
          name: "evaluation-detail",
          component: EvaluationDetail,
          meta: { title: "评估详情" },
        },
        {
          path: "live",
          name: "live",
          component: LiveSessionPage,
          meta: { role: "user", title: "实时跟练" },
        },
        {
          path: "live/list",
          name: "live-list",
          component: LiveSessionList,
          meta: { title: "会话记录" },
        },
        {
          path: "live/:id",
          name: "live-detail",
          component: LiveSessionDetail,
          meta: { title: "会话详情" },
        },
      ],
    },
  ],
});

// 统一做登录校验和角色跳转。
router.beforeEach((to) => {
  const authStore = useAuthStore();
  authStore.hydrate();

  if (to.meta.public) {
    return true;
  }

  if (!authStore.accessToken) {
    return { name: "login", query: { redirect: to.fullPath } };
  }

  if (to.name === "dashboard" && authStore.user?.role !== "admin") {
    return defaultRouteByRole(authStore.user?.role);
  }

  if (to.meta.role && authStore.user?.role !== to.meta.role) {
    return defaultRouteByRole(authStore.user?.role);
  }

  return true;
});

export default router;
