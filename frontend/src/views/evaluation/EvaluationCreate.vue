<script setup>
import { onMounted, reactive, ref } from "vue";
import { useRouter } from "vue-router";
import { ElMessage } from "element-plus";
import { getTemplateList } from "@/api/template";
import { createEvaluationTask } from "@/api/evaluation";

const router = useRouter();
const loading = ref(false);
const templates = ref([]);
const queryVideo = ref(null);

const form = reactive({
  template_id: "",
  task_name: "八段锦离线评估",
  score_scale: 100,
  hint_threshold: 0.18,
  hint_min_interval: 10,
  max_hints: 60,
});

async function loadTemplates() {
  const data = await getTemplateList();
  templates.value = data.filter((item) => item.status === "ready");
  if (!form.template_id && templates.value.length) {
    form.template_id = templates.value[0].id;
  }
}

function handleFileChange(file) {
  queryVideo.value = file.raw;
}

async function handleSubmit() {
  if (!queryVideo.value) {
    ElMessage.warning("请先上传待测视频");
    return;
  }

  const payload = new FormData();
  Object.entries(form).forEach(([key, value]) => payload.append(key, value));
  payload.append("query_video", queryVideo.value);

  loading.value = true;
  try {
    const result = await createEvaluationTask(payload);
    ElMessage.success("评估任务创建成功");
    router.push({ name: "evaluation-detail", params: { id: result.id } });
  } finally {
    loading.value = false;
  }
}

onMounted(loadTemplates);
</script>

<template>
  <div class="page-shell">
    <h2 class="page-title">离线评估</h2>

    <section class="soft-card page-panel">
      <h3 class="section-title">创建评估任务</h3>
      <el-form label-width="120px">
        <el-form-item label="任务名称">
          <el-input v-model="form.task_name" />
        </el-form-item>
        <el-form-item label="选择模板">
          <el-select v-model="form.template_id" style="width: 320px">
            <el-option
              v-for="item in templates"
              :key="item.id"
              :label="`${item.template_name} (${item.version})`"
              :value="item.id"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="得分上限">
          <el-input-number v-model="form.score_scale" :min="10" :max="100" />
        </el-form-item>
        <el-form-item label="提示阈值">
          <el-input-number
            v-model="form.hint_threshold"
            :min="0.01"
            :max="1"
            :step="0.01"
          />
        </el-form-item>
        <el-form-item label="提示间隔">
          <el-input-number v-model="form.hint_min_interval" :min="1" :max="100" />
        </el-form-item>
        <el-form-item label="提示上限">
          <el-input-number v-model="form.max_hints" :min="1" :max="9999" />
        </el-form-item>
        <el-form-item label="待测视频">
          <el-upload :auto-upload="false" :limit="1" :on-change="handleFileChange">
            <el-button type="primary" plain>选择视频文件</el-button>
          </el-upload>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="loading" @click="handleSubmit">
            提交评估
          </el-button>
        </el-form-item>
      </el-form>
    </section>
  </div>
</template>

<style scoped>
.page-panel {
  padding: 22px;
}
</style>
