<script setup>
import { computed } from "vue";

const props = defineProps({
  path: {
    type: String,
    default: "",
  },
});

const videoUrl = computed(() => {
  if (!props.path) {
    return "";
  }

  const normalized = props.path.replaceAll("\\", "/");
  const marker = "/media/";
  const index = normalized.indexOf(marker);

  if (index >= 0) {
    return normalized.slice(index);
  }

  if (normalized.startsWith("media/")) {
    return `/${normalized}`;
  }

  return normalized;
});
</script>

<template>
  <div class="video-player soft-card">
    <video
      v-if="videoUrl"
      :key="videoUrl"
      class="video-player__media"
      controls
      preload="metadata"
    >
      <source :src="videoUrl" type="video/mp4" />
      当前浏览器无法直接播放该视频，可尝试下载查看。
    </video>
    <div v-if="videoUrl" class="video-player__footer">
      <el-link :href="videoUrl" target="_blank" type="primary">打开视频</el-link>
    </div>
    <el-empty v-else description="暂无视频结果" />
  </div>
</template>

<style scoped>
.video-player {
  padding: 16px;
}

.video-player__media {
  width: 100%;
  border-radius: 14px;
  background: #0f172a;
}

.video-player__footer {
  margin-top: 10px;
}
</style>
