import request from "@/utils/request";

export function getTemplateList() {
  return request.get("/api/template-center/templates/");
}

export function getTemplateDetail(id) {
  return request.get(`/api/template-center/templates/${id}/`);
}

export function uploadTemplate(formData) {
  return request.post("/api/template-center/templates/upload/", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
}

export function buildTemplate(id) {
  return request.post(`/api/template-center/templates/${id}/build/`);
}

export function deleteTemplate(id) {
  return request.delete(`/api/template-center/templates/${id}/delete/`);
}
