<script setup>
import { onMounted, reactive } from "vue";
import { getTemplateList } from "@/api/template";
import { getEvaluationTasks } from "@/api/evaluation";
import { getLiveSessions } from "@/api/liveSession";
import StatCard from "@/components/StatCard.vue";

const stats = reactive({
  templates: 0,
  evaluations: 0,
  sessions: 0,
  latestScore: "--",
});

async function loadData() {
  const [templates, evaluations, sessions] = await Promise.all([
    getTemplateList(),
    getEvaluationTasks(),
    getLiveSessions(),
  ]);

  stats.templates = templates.length;
  stats.evaluations = evaluations.length;
  stats.sessions = sessions.length;
  stats.latestScore = evaluations[0]?.score ?? "--";
}

onMounted(loadData);
</script>

<template>
  <div class="page-shell">
    <h2 class="page-title">系统概览</h2>

    <section class="grid-three">
      <StatCard label="动作模板总数" :value="stats.templates" accent="#0f766e" />
      <StatCard label="离线评估任务" :value="stats.evaluations" accent="#f59e0b" />
      <StatCard label="实时跟练会话" :value="stats.sessions" accent="#1d4ed8" />
    </section>

    <section class="grid-two" style="margin-top: 20px">
      <div class="soft-card dashboard-panel">
        <h3 class="section-title">系统定位</h3>
        <p>
          当前系统围绕八段锦教学场景，支持模板视频管理、离线动作对比评估、分阶段结果查询与实时跟练会话管理。
        </p>
        <p>
          前端适合作为答辩展示界面，后续可以继续补统计图、实验结果页和用户训练档案。
        </p>
      </div>

      <div class="soft-card dashboard-panel">
        <h3 class="section-title">最近结果</h3>
        <div class="dashboard-score">{{ stats.latestScore }}</div>
        <div class="dashboard-score__label">最近一次离线评估得分</div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.dashboard-panel {
  min-height: 240px;
  padding: 22px;
}

.dashboard-panel p {
  color: #475569;
  line-height: 1.8;
}

.dashboard-score {
  margin-top: 24px;
  font-size: 64px;
  font-weight: 800;
  color: #0f766e;
}

.dashboard-score__label {
  margin-top: 10px;
  color: #64748b;
}
</style>
