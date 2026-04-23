import request from "@/utils/request";

export function getPoseModelOptions() {
  return request.get("/api/system/pose-model-options/");
}
