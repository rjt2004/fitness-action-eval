import request from "@/utils/request";

export function getLiveSessions() {
  return request.get("/api/live-session/sessions/");
}

export function startLiveSession(data) {
  return request.post("/api/live-session/sessions/start/", data);
}

export function getLiveSessionDetail(id) {
  return request.get(`/api/live-session/sessions/${id}/`);
}

export function stopLiveSession(id) {
  return request.post(`/api/live-session/sessions/${id}/stop/`);
}
