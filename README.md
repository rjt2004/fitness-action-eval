# 健身动作评价

本项目是一个“基于图像的健身动作评价系统设计与实现”毕业设计项目。系统以人体姿态估计为基础，通过标准动作模板与待评估动作之间的时序对齐、姿态相似度计算和局部偏差分析，实现健身动作评分、纠错提示、结果可视化和前后端管理。

当前系统以八段锦作为核心验证对象，已经完成算法专项化、Django 后端、Vue 前端、数据库设计、模板管理、离线评估、结果查询和实时跟练原型等功能。

## 当前功能

- 用户登录与角色权限：支持 `admin` 和 `user` 两类用户。
- 模板管理：管理员可上传标准视频、生成标准动作模板、重新生成或删除模板。
- 离线评估：用户上传待测视频，系统输出总分、分阶段结果、纠错提示和 DTW 可视化图。
- 可选视频导出：离线评估默认不导出对比视频，以提升长视频处理速度；需要留档时可手动开启。
- 实时跟练原型：支持摄像头编号、本地视频路径或网络视频流作为输入源，输出实时画面和提示。
- 结果中心：支持查看评估结果、提示列表、分阶段 DTW 图，并支持删除记录。
- 会话记录：支持查看实时跟练会话摘要、参数和提示列表，并支持删除记录。

## 技术栈

- 算法层：Python、OpenCV、MediaPipe Pose、NumPy、DTW
- 后端：Django、Django REST Framework、Simple JWT、MySQL
- 前端：Vue 3、Vite、Element Plus、Pinia、Axios、ECharts
- 数据库：MySQL
- 文件存储：本地 `media/` 目录

## 算法设计

### 姿态特征

系统首先从视频中提取人体 33 个关键点，并进行归一化处理：

- 以两髋中心作为坐标原点。
- 以肩中心到髋中心的距离作为尺度因子。
- 将人物位置、身高和画面尺度差异尽量消除。

每一帧的姿态特征由两部分组成：

- `66维`：33 个关键点的二维坐标。
- `8维`：左肩、右肩、左肘、右肘、左髋、右髋、左膝、右膝关节角。

最终形成 `74维` 姿态特征。

### 八段锦阶段划分

当前八段锦标准视频采用人工标注阶段边界：

- 预备势：`00:00-00:45`
- 双手托天理三焦：`00:45-02:20`
- 左右开弓似射雕：`02:20-03:45`
- 调理脾胃须单举：`03:45-05:05`
- 五劳七伤往后瞧：`05:05-06:10`
- 摇头摆尾去心火：`06:10-07:55`
- 两手攀足固肾腰：`07:55-10:00`
- 攒拳怒目增气力：`10:00-11:00`
- 背后七颠百病消：`11:00-11:50`
- 收势：`11:50-12:12`

系统会根据标准视频时间点生成阶段 ID，并在评分、提示和结果展示中使用。

### 评分规则

评分流程如下：

1. 提取标准视频和测试视频的人体姿态序列。
2. 构建 33 点坐标和 8 个关节角组成的特征向量。
3. 根据八段锦不同阶段对重点部位进行加权。
4. 使用带窗口约束的 DTW 对标准动作和测试动作进行时序对齐。
5. 计算对齐路径上的平均特征距离。
6. 将归一化距离映射为 `0-100` 分。

分数映射方式：

```python
score = 100 * (1 - normalized_distance / score_scale)
```

其中 `score_scale` 默认为 `8.0`，分数会被限制在 `0-100` 范围内。

### 提示规则

提示不是直接根据总分生成，而是根据局部身体部位误差生成。系统会分析：

- 头颈
- 肩肘
- 手部
- 躯干
- 腰胯
- 髋部
- 膝部
- 足部

每个部位误差由关键点误差和关节角误差融合：

```python
part_error = 0.8 * point_error + 0.2 * angle_error
```

不同八段锦阶段会有不同关注重点。例如：

- 双手托天理三焦：重点关注手部、肩部和躯干上拔。
- 左右开弓似射雕：重点关注手部开弓、肩背展开、马步和腰胯稳定。
- 调理脾胃须单举：重点关注上下撑按、肩部舒展和躯干中正。
- 五劳七伤往后瞧：重点关注头颈转动、肩部放松和躯干稳定。
- 两手攀足固肾腰：重点关注前俯幅度、手部下探和腰背放松。

当某个重点部位误差超过提示阈值时，系统会生成对应中文纠错建议。

## 系统模块

### 管理员功能

- 系统概览
- 模板管理
- 查看所有用户离线评估结果
- 查看所有用户实时会话记录

### 普通用户功能

- 离线评估
- 评估结果查看
- 实时跟练
- 实时会话记录

## 后端模块

后端位于 `backend/`：

- `apps/accounts`：用户、角色和 JWT 登录认证。
- `apps/template_manager`：标准视频、动作模板和文件资源管理。
- `apps/evaluation_center`：离线评估任务、结果入库、提示入库和删除。
- `apps/live_session`：实时跟练会话、实时预览帧、会话摘要和删除。
- `config`：Django 配置、统一响应、视频转码工具。

## 前端模块

前端位于 `frontend/`：

- `src/views/Login.vue`：登录页。
- `src/views/template/TemplateList.vue`：模板管理。
- `src/views/evaluation/EvaluationCreate.vue`：创建离线评估。
- `src/views/evaluation/EvaluationList.vue`：评估结果列表。
- `src/views/evaluation/EvaluationDetail.vue`：评估详情。
- `src/views/live/LiveSessionPage.vue`：实时跟练。
- `src/views/live/LiveSessionList.vue`：实时会话列表。
- `src/views/live/LiveSessionDetail.vue`：实时会话详情。

## 数据库核心表

- `sys_user`：用户表。
- `action_category`：动作类别表。
- `template_video`：标准模板表。
- `file_asset`：文件资源表。
- `evaluation_task`：离线评估任务表。
- `evaluation_phase_result`：分阶段结果表。
- `evaluation_hint`：纠错提示表。
- `live_session`：实时跟练会话表。

## 本地运行

### 1. 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

前端依赖：

```bash
cd frontend
npm install
```

### 2. 配置数据库

后端默认从 `backend/.env` 读取数据库配置。示例文件见：

```text
backend/.env.example
```

执行迁移：

```bash
cd backend
python manage.py migrate
```

创建管理员：

```bash
python manage.py createsuperuser
```

### 3. 启动后端

```bash
cd backend
python manage.py runserver
```

局域网访问时使用：

```bash
python manage.py runserver 0.0.0.0:8000
```

### 4. 启动前端

```bash
cd frontend
npm run dev
```

浏览器访问：

```text
http://127.0.0.1:5173
```

局域网其他设备访问时，将 `127.0.0.1` 替换为本机局域网 IP。

## 常用参数

离线评估默认参数：

- `frame_stride=6`
- `smooth_window=3`
- `hint_threshold=0.20`
- `hint_min_interval=12`
- `max_hints=24`
- `export_video=false`

实时跟练默认参数：

- `camera_width=640`
- `camera_height=360`
- `frame_stride=4`
- `smooth_window=5`
- `hint_threshold=0.18`
- `hint_min_interval=8`
- `max_hints=40`
- `ref_search_window=20`
- `export_video=false`

说明：

- `frame_stride` 越大，处理越快，但评分粒度会降低。
- `smooth_window` 越大，姿态更平滑，但响应更慢。
- `hint_threshold` 越低，提示越容易触发。
- `hint_min_interval` 越大，提示越稀疏。
- `max_hints` 控制最多输出多少条提示。

## 长视频性能建议

对于 12 分钟左右的八段锦视频，建议：

- 默认不导出对比视频，只生成评分、提示和 DTW 图。
- 如果需要展示或留档，再手动开启“导出对比视频”。
- 常规评估使用 `frame_stride=6`。
- 更快评估可尝试 `frame_stride=8` 或 `frame_stride=10`。
- 正式实验对比时必须固定 `frame_stride` 和 `smooth_window`，否则分数不可直接横向比较。

## 实时跟练说明

当前实时跟练是原型功能，适合使用本机摄像头、本地视频文件或 OpenCV 可读取的网络视频流进行测试。

注意：

- `camera_source=0` 表示运行后端这台电脑的默认摄像头。
- 如果前端在另一台电脑访问，当前后端仍然读取服务器电脑的视频源。
- 若要使用浏览器所在电脑的摄像头，需要进一步实现浏览器采集视频帧并通过 WebSocket/WebRTC 上传到后端。

## Git 提交建议

通常不要提交以下本地文件：

- `.idea/workspace.xml`
- `frontend/.vite/`
- `media/`
- `input/`
- `output/`
- 临时实验产物

## 当前状态

当前项目已经具备毕业设计展示所需的主要闭环：

1. 标准视频模板管理。
2. 八段锦专项评分规则。
3. 离线评估与结果查询。
4. 分阶段 DTW 可视化。
5. 中文纠错提示。
6. 实时跟练原型。
7. Django + Vue 前后端系统。
8. MySQL 数据库存储。

后续建议重点补充实验数据、评分结果统计、论文系统设计说明和实验分析，而不是继续大规模扩展功能。
