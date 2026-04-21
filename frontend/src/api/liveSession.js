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

export function pauseLiveSession(id) {
  return request.post(`/api/live-session/sessions/${id}/pause/`);
}

export function resumeLiveSession(id) {
  return request.post(`/api/live-session/sessions/${id}/resume/`);
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
