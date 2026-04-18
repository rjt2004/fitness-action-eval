import axios from "axios";
import { ElMessage } from "element-plus";
import router from "@/router";
import { useAuthStore } from "@/stores/auth";

const request = axios.create({
  baseURL: "/",
  timeout: 120000,
});

request.interceptors.request.use((config) => {
  const authStore = useAuthStore();
  authStore.hydrate();

  if (authStore.accessToken) {
    config.headers.Authorization = `Bearer ${authStore.accessToken}`;
  }

  return config;
});

request.interceptors.response.use(
  (response) => {
    const payload = response.data;

    if (payload && typeof payload === "object" && "code" in payload) {
      if (payload.code === 0) {
        return payload.data;
      }

      ElMessage.error(payload.message || "请求失败");
      return Promise.reject(new Error(payload.message || "Request failed"));
    }

    return payload;
  },
  (error) => {
    const authStore = useAuthStore();
    const responseData = error.response?.data || {};

    if (error.response?.status === 401) {
      authStore.clearAuth();
      if (router.currentRoute.value.name !== "login") {
        router.push({ name: "login" });
      }
    }

    const firstValidationMessage =
      responseData?.data?.non_field_errors?.[0] ||
      responseData?.data?.password?.[0] ||
      responseData?.data?.username?.[0];

    ElMessage.error(
      responseData?.message ||
        responseData?.detail ||
        firstValidationMessage ||
        error.message ||
        "网络异常",
    );
    return Promise.reject(error);
  },
);

export default request;
