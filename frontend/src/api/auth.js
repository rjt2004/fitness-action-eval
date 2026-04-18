import request from "@/utils/request";

export function login(data) {
  return request.post("/api/auth/login/", data);
}

export function refreshToken(data) {
  return request.post("/api/auth/refresh/", data);
}

export function getCurrentUser() {
  return request.get("/api/auth/me/");
}
