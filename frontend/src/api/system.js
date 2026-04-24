import request from "@/utils/request";

// 系统公共元数据接口。
export function getPoseModelOptions() {
  return request.get("/api/system/pose-model-options/");
}
