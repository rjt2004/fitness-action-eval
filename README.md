# 基于图像的健身动作评价系统

本项目是一个面向健身动作教学与纠错场景的毕业设计系统。系统以人体姿态估计为基础，结合标准动作模板、DTW 时序对齐、关键点与关节角特征、局部部位误差分析，实现离线动作评估、实时跟练提示、模板管理、结果查询和用户管理等功能。当前验证动作以八段锦为主。

## 功能概览

- 用户登录与权限管理：支持管理员和普通用户两类角色。
- 模板管理：管理员上传标准动作视频，生成动作模板和阶段规则配置。
- 离线评估：上传待测视频后，系统输出总分、阶段得分、部位得分、纠错提示、DTW 图和可选对比视频。
- 实时跟练：支持摄像头、本地视频或 OpenCV 可读取的视频流作为输入，实时显示骨架画面、参考视频和纠错提示。
- 结果中心：查看、删除离线评估任务和实时跟练会话记录。
- 算法模块：完成人体关键点提取、特征归一化、平滑处理、DTW 对齐、评分映射和提示生成。

## 技术栈

- 算法：Python、OpenCV、MediaPipe Pose、NumPy、Matplotlib、Pillow、ImageIO
- 后端：Django、Django REST Framework、Simple JWT、MySQL、PyMySQL
- 前端：Vue 3、Vite、Element Plus、Pinia、Axios、ECharts
- 文件存储：本地 `media/` 目录

## 目录结构

```text
fitness-action-eval/
├─ backend/                 # Django 后端
│  ├─ apps/                 # 业务应用：用户、模板、评估、实时跟练
│  ├─ config/               # 后端配置、路由、视频工具
│  ├─ .env.example          # 后端环境变量示例
│  └─ manage.py
├─ fitness_action_eval/     # 姿态识别、DTW、评分、可视化等算法代码
├─ frontend/                # Vue 前端
├─ media/                   # 上传文件与运行结果，运行时自动生成，不提交
├─ input/                   # 本地测试输入目录，运行时使用
├─ output/                  # 本地测试输出目录，运行时使用
├─ pose_landmarker_lite.task
├─ pose_landmarker_full.task
├─ pose_landmarker_heavy.task
├─ requirements.txt
└─ README.md
```

## 环境要求

建议使用以下版本，换设备部署时尽量保持一致：

- Python 3.10 或 3.11
- Node.js 18 或 20
- MySQL 8.0
- Windows 10/11、macOS 或 Linux 均可运行；实时摄像头功能需要运行后端的设备能访问摄像头

说明：项目已包含 `pose_landmarker_lite.task`、`pose_landmarker_full.task` 和 `pose_landmarker_heavy.task` 三个 MediaPipe 姿态模型文件，克隆项目后请确认这三个文件位于项目根目录。

## 后端部署

### 1. 克隆项目

```bash
git clone git@github.com:rjt2004/fitness-action-eval.git
cd fitness-action-eval
```

如果没有配置 SSH，也可以使用 HTTPS 地址：

```bash
git clone https://github.com/rjt2004/fitness-action-eval.git
cd fitness-action-eval
```

### 2. 创建并激活 Python 虚拟环境

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. 创建 MySQL 数据库

登录 MySQL 后执行：

```sql
CREATE DATABASE fitness_action_eval DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 4. 配置后端环境变量

复制示例配置文件：

Windows PowerShell：

```powershell
Copy-Item backend\.env.example backend\.env
```

macOS/Linux：

```bash
cp backend/.env.example backend/.env
```

根据本机数据库账号修改 `backend/.env`：

```text
DJANGO_SECRET_KEY=replace-with-your-secret-key
DJANGO_DEBUG=true
DJANGO_ALLOWED_HOSTS=127.0.0.1,localhost
DJANGO_TIME_ZONE=Asia/Shanghai

MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_DATABASE=fitness_action_eval
MYSQL_USER=root
MYSQL_PASSWORD=123456
```

如果需要让局域网其他设备访问后端，把本机局域网 IP 加入 `DJANGO_ALLOWED_HOSTS`，例如：

```text
DJANGO_ALLOWED_HOSTS=127.0.0.1,localhost,192.168.1.10
```

### 5. 初始化数据库

```bash
cd backend
python manage.py migrate
python manage.py createsuperuser
```

创建的超级用户可用于登录系统，管理员登录后可以继续创建普通用户。

### 6. 启动后端

本机访问：

```bash
python manage.py runserver 127.0.0.1:8000
```

局域网访问：

```bash
python manage.py runserver 0.0.0.0:8000
```

后端健康检查接口：

```text
http://127.0.0.1:8000/api/system/health/
```

## 前端部署

新开一个终端进入项目根目录，安装并启动前端：

```bash
cd frontend
npm install
npm run dev
```

浏览器访问：

```text
http://127.0.0.1:5173
```

开发环境下，Vite 会把 `/api` 和 `/media` 请求代理到 `http://127.0.0.1:8000`。如果后端不在本机运行，请修改 `frontend/vite.config.js` 中的代理地址。

前端生产构建：

```bash
cd frontend
npm run build
```

## 首次使用流程

1. 启动 MySQL、Django 后端和 Vue 前端。
2. 使用 `createsuperuser` 创建的管理员账号登录。
3. 在模板管理页面上传标准动作视频，并选择姿态模型生成模板。
4. 普通用户或管理员进入离线评估页面，选择模板并上传待测视频。
5. 在评估详情页查看总分、阶段得分、部位得分、纠错提示、DTW 图和可选对比视频。
6. 在实时跟练页面选择模板和输入源，开始实时跟练会话。

## 常用参数说明

离线评估常用参数：

- `frame_stride`：抽帧步长，值越大处理越快，但评分粒度会降低。
- `smooth_window`：姿态序列平滑窗口，值越大越平滑，但响应会变慢。
- `hint_threshold`：提示触发阈值，值越低越容易生成纠错提示。
- `hint_min_interval`：提示最小间隔，用于避免短时间内重复提示。
- `max_hints`：单次评估最多保留的提示数量。
- `export_video`：是否导出标准动作与待测动作的对比视频。

实时跟练常用参数：

- `camera_source`：输入源，`0` 表示运行后端设备的默认摄像头，也可以填写本地视频路径或网络视频流地址。
- `camera_width`、`camera_height`：摄像头采集分辨率。
- `frame_stride`：实时处理抽帧步长。
- `ref_search_window`：参考帧局部搜索窗口，用于实时跟练时匹配当前动作进度。

## 注意事项

- `media/`、`input/`、`output/`、`frontend/node_modules/` 和 `frontend/dist/` 为运行时或构建产物，不需要提交到 Git。
- 实时摄像头输入由后端进程读取；如果前端在另一台设备访问，摄像头仍然是后端所在设备的摄像头。
- 视频导出依赖 `imageio-ffmpeg` 提供的 FFmpeg，可减少不同设备上手动安装 FFmpeg 的步骤。
- MediaPipe 对 Python 版本较敏感，推荐使用 Python 3.10 或 3.11。
- 如果安装依赖时 OpenCV 或 MediaPipe 下载较慢，可以切换到稳定的 PyPI 镜像源后重试。

## 基础检查命令

后端检查：

```bash
cd backend
python manage.py check
```

前端构建检查：

```bash
cd frontend
npm run build
```
