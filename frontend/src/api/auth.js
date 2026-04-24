import request from "@/utils/request";

// 认证相关接口。
export function login(data) {
  return request.post("/api/auth/login/", data);
}

export function refreshToken(data) {
  return request.post("/api/auth/refresh/", data);
}

export function getCurrentUser() {
  return request.get("/api/auth/me/");
}
