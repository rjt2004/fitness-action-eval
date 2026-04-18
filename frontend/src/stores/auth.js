import { defineStore } from "pinia";

const STORAGE_KEY = "fitness_action_eval_auth";

export const useAuthStore = defineStore("auth", {
  state: () => ({
    accessToken: "",
    refreshToken: "",
    user: null,
    initialized: false,
  }),
  actions: {
    hydrate() {
      if (this.initialized) {
        return;
      }
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (raw) {
        try {
          const parsed = JSON.parse(raw);
          this.accessToken = parsed.accessToken || "";
          this.refreshToken = parsed.refreshToken || "";
          this.user = parsed.user || null;
        } catch (error) {
          window.localStorage.removeItem(STORAGE_KEY);
        }
      }
      this.initialized = true;
    },
    setAuth(payload) {
      this.accessToken = payload.access || "";
      this.refreshToken = payload.refresh || "";
      this.user = payload.user || null;
      this.initialized = true;
      window.localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          accessToken: this.accessToken,
          refreshToken: this.refreshToken,
          user: this.user,
        }),
      );
    },
    clearAuth() {
      this.accessToken = "";
      this.refreshToken = "";
      this.user = null;
      this.initialized = true;
      window.localStorage.removeItem(STORAGE_KEY);
    },
  },
});
