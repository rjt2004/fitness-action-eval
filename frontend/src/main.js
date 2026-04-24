import { createApp } from "vue";
import { createPinia } from "pinia";
import ElementPlus from "element-plus";
import * as ElementPlusIconsVue from "@element-plus/icons-vue";

import "element-plus/dist/index.css";
import "./styles.css";

import App from "./App.vue";
import router from "./router";

// 前端应用入口：注册 Pinia、路由和 Element Plus 组件库。
const app = createApp(App);
const pinia = createPinia();

Object.entries(ElementPlusIconsVue).forEach(([key, component]) => {
  app.component(key, component);
});

app.use(pinia);
app.use(router);
app.use(ElementPlus);
app.mount("#app");
