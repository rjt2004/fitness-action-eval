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

export function getAdminUsers() {
  return request.get("/api/auth/users/");
}

export function createAdminUser(data) {
  return request.post("/api/auth/users/", data);
}

export function updateAdminUser(userId, data) {
  return request.patch(`/api/auth/users/${userId}/`, data);
}

export function resetAdminUserPassword(userId, data) {
  return request.post(`/api/auth/users/${userId}/reset-password/`, data);
}

export function deleteAdminUser(userId) {
  return request.delete(`/api/auth/users/${userId}/`);
}
