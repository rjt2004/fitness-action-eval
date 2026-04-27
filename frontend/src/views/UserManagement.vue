<script setup>
import { computed, onMounted, reactive, ref } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import { useAuthStore } from "@/stores/auth";
import {
  createAdminUser,
  deleteAdminUser,
  getAdminUsers,
  resetAdminUserPassword,
  updateAdminUser,
} from "@/api/auth";

const authStore = useAuthStore();
const loading = ref(false);
const users = ref([]);

const filter = reactive({
  keyword: "",
  role: "",
});

const editDialogVisible = ref(false);
const passwordDialogVisible = ref(false);
const dialogMode = ref("create");
const currentRow = ref(null);

const editForm = reactive({
  username: "",
  password: "",
  real_name: "",
  phone: "",
  email: "",
  role: "user",
  is_active: true,
});

const passwordForm = reactive({
  new_password: "",
});

const filteredUsers = computed(() => {
  const keyword = filter.keyword.trim().toLowerCase();
  return users.value.filter((item) => {
    const roleMatched = !filter.role || item.role === filter.role;
    const keywordMatched =
      !keyword ||
      [item.username, item.real_name, item.phone, item.email]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(keyword));
    return roleMatched && keywordMatched;
  });
});

function roleLabel(role) {
  return role === "admin" ? "管理员" : "普通用户";
}

function resetEditForm() {
  editForm.username = "";
  editForm.password = "";
  editForm.real_name = "";
  editForm.phone = "";
  editForm.email = "";
  editForm.role = "user";
  editForm.is_active = true;
}

async function loadData() {
  loading.value = true;
  try {
    users.value = await getAdminUsers();
  } finally {
    loading.value = false;
  }
}

function openCreateDialog() {
  dialogMode.value = "create";
  currentRow.value = null;
  resetEditForm();
  editDialogVisible.value = true;
}

function openEditDialog(row) {
  dialogMode.value = "edit";
  currentRow.value = row;
  resetEditForm();
  editForm.username = row.username || "";
  editForm.real_name = row.real_name || "";
  editForm.phone = row.phone || "";
  editForm.email = row.email || "";
  editForm.role = row.role || "user";
  editForm.is_active = Boolean(row.is_active);
  editDialogVisible.value = true;
}

async function submitEditForm() {
  if (dialogMode.value === "create") {
    await createAdminUser({
      username: editForm.username,
      password: editForm.password,
      real_name: editForm.real_name,
      phone: editForm.phone,
      email: editForm.email,
      role: editForm.role,
      is_active: editForm.is_active,
    });
    ElMessage.success("用户创建成功");
  } else if (currentRow.value) {
    await updateAdminUser(currentRow.value.id, {
      real_name: editForm.real_name,
      phone: editForm.phone,
      email: editForm.email,
      role: editForm.role,
      is_active: editForm.is_active,
    });
    ElMessage.success("用户信息已更新");
  }
  editDialogVisible.value = false;
  await loadData();
}

function openResetPasswordDialog(row) {
  currentRow.value = row;
  passwordForm.new_password = "";
  passwordDialogVisible.value = true;
}

async function submitPasswordForm() {
  if (!currentRow.value) return;
  await resetAdminUserPassword(currentRow.value.id, {
    new_password: passwordForm.new_password,
  });
  passwordDialogVisible.value = false;
  ElMessage.success("密码重置成功");
}

async function handleDelete(row) {
  await ElMessageBox.confirm(`确认删除用户“${row.username}”吗？`, "删除确认", {
    type: "warning",
  });
  await deleteAdminUser(row.id);
  ElMessage.success("用户已删除");
  await loadData();
}

async function toggleUserStatus(row) {
  await updateAdminUser(row.id, { is_active: !row.is_active });
  ElMessage.success(row.is_active ? "用户已禁用" : "用户已启用");
  await loadData();
}

function formatDate(value) {
  if (!value) return "--";
  return String(value).replace("T", " ").slice(0, 19);
}

onMounted(loadData);
</script>

<template>
  <div class="page-shell">
    <div class="page-head">
      <h2 class="page-title">用户管理</h2>
      <el-button type="primary" @click="openCreateDialog">新建用户</el-button>
    </div>

    <section class="soft-card page-panel">
      <div class="toolbar">
        <el-input
          v-model="filter.keyword"
          placeholder="搜索用户名、姓名、手机号或邮箱"
          clearable
          class="toolbar__search"
        />
        <el-select v-model="filter.role" placeholder="全部角色" clearable class="toolbar__select">
          <el-option label="管理员" value="admin" />
          <el-option label="普通用户" value="user" />
        </el-select>
      </div>

      <el-table v-loading="loading" :data="filteredUsers" stripe style="width: 100%">
        <el-table-column prop="username" label="用户名" min-width="140" />
        <el-table-column prop="real_name" label="姓名" min-width="120" />
        <el-table-column prop="phone" label="手机号" min-width="140" />
        <el-table-column prop="email" label="邮箱" min-width="180" />
        <el-table-column label="角色" width="110">
          <template #default="{ row }">
            <el-tag :type="row.role === 'admin' ? 'danger' : 'info'" effect="light">
              {{ roleLabel(row.role) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="状态" width="110">
          <template #default="{ row }">
            <el-tag :type="row.is_active ? 'success' : 'warning'" effect="light">
              {{ row.is_active ? "启用" : "禁用" }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="最近登录" min-width="170">
          <template #default="{ row }">{{ formatDate(row.last_login) }}</template>
        </el-table-column>
        <el-table-column label="创建时间" min-width="170">
          <template #default="{ row }">{{ formatDate(row.date_joined) }}</template>
        </el-table-column>
        <el-table-column label="操作" min-width="320" fixed="right">
          <template #default="{ row }">
            <div class="action-row">
              <el-button size="small" type="primary" plain @click="openEditDialog(row)">编辑</el-button>
              <el-button size="small" plain @click="openResetPasswordDialog(row)">重置密码</el-button>
              <el-button
                size="small"
                :type="row.is_active ? 'warning' : 'success'"
                plain
                :disabled="row.id === authStore.user?.id"
                @click="toggleUserStatus(row)"
              >
                {{ row.is_active ? "禁用" : "启用" }}
              </el-button>
              <el-button
                size="small"
                type="danger"
                plain
                :disabled="row.id === authStore.user?.id"
                @click="handleDelete(row)"
              >
                删除
              </el-button>
            </div>
          </template>
        </el-table-column>
      </el-table>
    </section>

    <el-dialog
      v-model="editDialogVisible"
      :title="dialogMode === 'create' ? '新建用户' : '编辑用户'"
      width="520px"
    >
      <el-form label-width="90px" class="dialog-form">
        <el-form-item label="用户名">
          <el-input v-model="editForm.username" :disabled="dialogMode === 'edit'" />
        </el-form-item>
        <el-form-item v-if="dialogMode === 'create'" label="初始密码">
          <el-input v-model="editForm.password" type="password" show-password />
        </el-form-item>
        <el-form-item label="姓名">
          <el-input v-model="editForm.real_name" />
        </el-form-item>
        <el-form-item label="手机号">
          <el-input v-model="editForm.phone" />
        </el-form-item>
        <el-form-item label="邮箱">
          <el-input v-model="editForm.email" />
        </el-form-item>
        <el-form-item label="角色">
          <el-select v-model="editForm.role" style="width: 100%">
            <el-option label="管理员" value="admin" />
            <el-option label="普通用户" value="user" />
          </el-select>
        </el-form-item>
        <el-form-item label="启用状态">
          <el-switch v-model="editForm.is_active" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="editDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="submitEditForm">
          {{ dialogMode === "create" ? "创建" : "保存" }}
        </el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="passwordDialogVisible" title="重置密码" width="420px">
      <el-form label-width="90px" class="dialog-form">
        <el-form-item label="新密码">
          <el-input v-model="passwordForm.new_password" type="password" show-password />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="passwordDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="submitPasswordForm">确认重置</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.page-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}

.page-panel {
  padding: 22px;
}

.toolbar {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
}

.toolbar__search {
  max-width: 360px;
}

.toolbar__select {
  width: 160px;
}

.action-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.dialog-form {
  padding-right: 12px;
}
</style>
