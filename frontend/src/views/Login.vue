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
    <main class="login-main">
      <h1 class="login-title">基于图像的健身动作评价系统</h1>
      <div class="login-panel">
        <h2>用户登录</h2>
        <el-form label-position="top" @submit.prevent="handleLogin">
          <el-form-item label="用户名">
            <el-input v-model="form.username" placeholder="请输入用户名" size="large" />
          </el-form-item>
          <el-form-item label="密码">
            <el-input
              v-model="form.password"
              type="password"
              placeholder="请输入密码"
              show-password
              size="large"
            />
          </el-form-item>
          <el-button
            type="primary"
            size="large"
            native-type="submit"
            class="login-panel__submit"
            :loading="loading"
          >
            登录系统
          </el-button>
        </el-form>
      </div>
    </main>
  </div>
</template>

<style scoped>
.login-page {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background:
    linear-gradient(180deg, #f3f7f5 0%, #edf3f1 100%);
}

.login-main {
  display: flex;
  align-items: center;
  flex-direction: column;
  justify-content: center;
  width: 100%;
  padding: 48px 24px;
}

.login-title {
  width: min(440px, 100%);
  margin: 0 0 28px;
  color: #0f3f3b;
  font-size: 32px;
  line-height: 1.35;
  text-align: center;
  font-weight: 800;
}

.login-panel {
  width: min(440px, 100%);
  padding: 38px 40px 40px;
  border: 1px solid rgba(15, 118, 110, 0.14);
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.96);
  box-shadow: 0 18px 48px rgba(15, 94, 89, 0.12);
}

.login-panel h2 {
  margin: 0 0 26px;
  color: #102f2c;
  font-size: 24px;
  line-height: 1.3;
  text-align: center;
}

.login-panel__submit {
  width: 100%;
  margin-top: 6px;
}

:deep(.el-form-item__label) {
  color: #193f3b;
  font-weight: 600;
}

:deep(.el-input__wrapper) {
  border-radius: 8px;
  box-shadow: 0 0 0 1px rgba(15, 118, 110, 0.2) inset;
}

:deep(.el-input__wrapper.is-focus) {
  box-shadow: 0 0 0 1px #0f766e inset;
}

:deep(.el-button--primary) {
  --el-button-bg-color: #0f766e;
  --el-button-border-color: #0f766e;
  --el-button-hover-bg-color: #115e59;
  --el-button-hover-border-color: #115e59;
  --el-button-active-bg-color: #134e4a;
  --el-button-active-border-color: #134e4a;
}

@media (max-width: 900px) {
  .login-main {
    padding: 32px 18px;
  }

  .login-title {
    font-size: 26px;
  }

  .login-panel {
    padding: 30px 24px 32px;
  }
}
</style>
