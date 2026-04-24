import request from "@/utils/request";

// 离线评估接口。
export function getEvaluationTasks() {
  return request.get("/api/evaluation-center/tasks/");
}

export function createEvaluationTask(formData) {
  return request.post("/api/evaluation-center/tasks/create/", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
}

export function getEvaluationTaskDetail(id) {
  return request.get(`/api/evaluation-center/tasks/${id}/`);
}

export function getEvaluationTaskPhases(id) {
  return request.get(`/api/evaluation-center/tasks/${id}/phases/`);
}

export function getEvaluationTaskHints(id) {
  return request.get(`/api/evaluation-center/tasks/${id}/hints/`);
}

export function deleteEvaluationTask(id) {
  return request.delete(`/api/evaluation-center/tasks/${id}/delete/`);
}
