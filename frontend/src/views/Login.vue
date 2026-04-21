<script setup>
import { reactive, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import { ElMessage } from "element-plus";
import { login } from "@/api/auth";
import { useAuthStore } from "@/stores/auth";

const router = useRouter();
const route = useRoute();
const authStore = useAuthStore();
const loading = ref(false);

const form = reactive({
  username: "",
  password: "",
});

async function handleLogin() {
  loading.value = true;
  try {
    const data = await login(form);
    authStore.setAuth(data);
    ElMessage.success("登录成功");
    if (route.query.redirect) {
      router.push(route.query.redirect);
      return;
    }
    router.push(data.user?.role === "admin" ? "/" : "/evaluation/create");
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <div class="login-page">
    <div class="login-page__hero">
      <div class="login-page__badge">Fitness Action Evaluation</div>
      <h1>基于图像的健身动作评价系统</h1>
    </div>

    <div class="login-panel soft-card">
      <h2>用户登录</h2>
      <el-form label-position="top" @submit.prevent="handleLogin">
        <el-form-item label="用户名">
          <el-input v-model="form.username" placeholder="请输入用户名" />
        </el-form-item>
        <el-form-item label="密码">
          <el-input
            v-model="form.password"
            type="password"
            placeholder="请输入密码"
            show-password
          />
        </el-form-item>
        <el-button
          type="primary"
          size="large"
          native-type="submit"
          style="width: 100%"
          :loading="loading"
        >
          登录系统
        </el-button>
      </el-form>
    </div>
  </div>
</template>

<style scoped>
.login-page {
  display: grid;
  grid-template-columns: 1.2fr 0.9fr;
  min-height: 100vh;
  padding: 40px;
  gap: 32px;
  background:
    linear-gradient(135deg, rgba(15, 118, 110, 0.14), transparent 35%),
    radial-gradient(circle at right top, rgba(245, 158, 11, 0.16), transparent 24%);
}

.login-page__hero {
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 48px;
}

.login-page__badge {
  display: inline-flex;
  width: fit-content;
  padding: 10px 16px;
  border-radius: 999px;
  color: #115e59;
  background: rgba(255, 255, 255, 0.7);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.login-page__hero h1 {
  margin: 20px 0 12px;
  font-size: 54px;
  line-height: 1.1;
}

.login-page__hero p {
  max-width: 620px;
  color: #475569;
  font-size: 18px;
  line-height: 1.8;
}

.login-panel {
  align-self: center;
  padding: 32px;
}

.login-panel h2 {
  margin: 0 0 20px;
}
</style>
