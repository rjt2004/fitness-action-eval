import request from "@/utils/request";

// 实时跟练接口。
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

export function deleteLiveSession(id) {
  return request.delete(`/api/live-session/sessions/${id}/delete/`);
}

export function getLiveSessionPreviewFrame(id) {
  return request.get(`/api/live-session/sessions/${id}/preview-frame/`, {
    responseType: "blob",
  });
}
