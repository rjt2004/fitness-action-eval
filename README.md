# 基于图像的健身动作评分系统

## 项目简介

本项目面向“基于图像的健身动作评分系统设计”课题，当前已完成一个基于人体姿态估计与 DTW（动态时间规整）的动作评分原型。系统输入一段标准动作视频和一段待测动作视频，先提取人体 33 个关键点，再对两段动作序列进行时序对齐，最终输出动作得分、局部偏差提示以及可视化结果。

当前实现已针对八段锦做了专项适配：在原有姿态提取和 DTW 对齐基础上，加入了八段锦 10 个环节（预备势、八式、收势）的阶段划分、关键关节角特征、分动作部位加权以及中文实时纠错提示，适合用于中期答辩阶段展示算法设计、系统流程和阶段性实验结果。

## 目前已完成的工作

1. 完成了基于 MediaPipe Pose 的视频人体姿态关键点提取。
2. 实现了多人场景下的目标人物筛选，优先跟踪画面中心且时序连续的人体。
3. 实现了关键点归一化处理，降低人物位置、尺度差异对评分结果的影响。
4. 实现了姿态时序特征平滑与标准化，提升动作序列稳定性。
5. 实现了基于 DTW 的动作时序对齐，可处理动作速度不一致的问题。
6. 实现了 0-100 分的动作评分映射方法，可直观给出动作质量分数。
7. 实现了基于八段锦动作阶段的反馈提示生成，可定位肩、手、腰胯、膝等关键偏差并输出中文纠错建议。
8. 实现了模板导出、结果导出和对比视频生成功能，可输出 JSON 评分结果、对齐曲线图和叠加提示的视频。
9. 新增实时摄像头/视频跟练模式，支持按参数设置摄像头编号、分辨率、镜像、输出视频和摘要 JSON。

## 系统流程

1. 读取标准动作视频与待评分视频。
2. 使用 MediaPipe Pose 提取每一帧的人体 33 关键点。
3. 对关键点进行归一化、平滑和特征标准化处理。
4. 使用 DTW 对标准动作序列与待测动作序列进行时序对齐。
5. 根据归一化 DTW 距离计算最终得分。
6. 计算局部身体部位误差并生成动作纠正提示。
7. 输出 JSON、曲线图和可视化反馈视频。

## 主要文件说明

- `demo_dtw_scoring.py`：基于参考视频与测试视频的八段锦评分入口。
- `demo_camera_coach.py`：基于模板的实时摄像头/视频跟练入口。
- `fitness_action_eval/pose.py`：姿态提取、目标人物筛选与实时预览。
- `fitness_action_eval/dtw.py`：带约束窗口的 DTW 对齐与分数映射。
- `fitness_action_eval/feedback.py`：八段锦局部误差分析与阶段性提示生成。
- `fitness_action_eval/baduanjin.py`：八段锦动作阶段、关键部位、关节角与提示规则。
- `fitness_action_eval/visualization.py`：骨架绘制、曲线图输出、视频叠加与实时显示。
- `fitness_action_eval/pipeline.py`：总流程编排模块。
- `input/`：输入测试视频目录。
- `output/`：评分结果、图像和视频输出目录。
- `pose_landmarker_full.task` 等：MediaPipe 姿态估计模型文件。
- `requirements.txt`：项目依赖列表。

## 运行环境

建议使用 Python 3.10 及以上版本。

安装依赖：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 运行示例

```bash
python export_pose_template.py ^
  --ref_video input\baduan.mp4 ^
  --template_path output\baduanjin_template.npz ^
  --frame_stride 4
```

```bash
python demo_template_scoring.py ^
  --template_path output\baduanjin_template.npz ^
  --query_video input\baduan_wrong.mp4 ^
  --out_json output\baduan_wrong_baduanjin_result.json ^
  --out_plot output\baduan_wrong_baduanjin_plot.png ^
  --out_video output\baduan_wrong_baduanjin_overlay.mp4
```

```bash
python demo_camera_coach.py ^
  --template_path output\baduanjin_template.npz ^
  --ref_video input\baduan.mp4 ^
  --camera_source 0 ^
  --camera_width 1280 ^
  --camera_height 720 ^
  --camera_mirror ^
  --out_video output\baduanjin_camera_overlay.mp4 ^
  --preview
```

## 输入与输出说明

输入参数：

- `--ref_video`：标准动作参考视频。
- `--query_video`：待评分视频。
- `--template_path`：已导出的标准动作模板。
- `--camera_source`：摄像头编号或视频路径，`0` 表示默认摄像头。
- `--out_json`：保存最终得分、DTW 距离和动作提示信息。
- `--out_plot`：保存 DTW 对齐可视化曲线图。
- `--out_video`：保存左右对比版评分视频。
- `--preview`：实时显示处理过程和最终叠加效果，按 `q` 可关闭窗口。
- `--frame_stride`：长视频抽帧步长，八段锦建议设置为 `3~5`。
- `--camera_width` / `--camera_height`：实时跟练模式下的摄像头分辨率。
- `--camera_mirror`：实时跟练模式下是否镜像显示。

常用默认配置：

- 姿态模型默认 `pose_landmarker_lite.task`
- 单帧检测人数默认 `1`
- 时间平滑窗口默认 `7`
- 八段锦长视频默认抽帧步长 `4`
- 评分映射参数默认 `8.0`
- 提示触发阈值默认 `0.18`
- 提示最小间隔默认 `8`
- 最大提示数量默认 `40`

## 当前结果特点

1. 能够对连续动作视频进行整体评分。
2. 能够适应不同动作节奏下的序列对齐，并针对长视频使用带约束窗口的 DTW 提高效率。
3. 能够输出带动作语义的局部偏差提示，而不仅仅是单一总分。
4. 已完成模块化拆分，便于后续扩展、调试与答辩展示。
5. 已支持左右双窗口对比预览、对比版结果视频导出以及实时摄像头跟练。
6. 已具备中期答辩演示所需的基本输入、处理、输出闭环。

## 实时可视化说明

运行时增加 `--preview` 参数即可实时查看左右双窗口对比效果：

1. 左侧显示参考动作视频。
2. 右侧显示待评分动作视频。
3. 画面中同步显示当前 DTW 对齐进度、局部误差和动作提示。

示例：

```bash
python demo_dtw_scoring.py ^
  --ref_video input\situp_test.mp4 ^
  --query_video input\situp_test_wrong.mp4 ^
  --out_json output\dtw_situp_result.json ^
  --out_plot output\dtw_situp_plot.png ^
  --out_video output\dtw_situp_overlay.mp4 ^
  --preview
```

## 下一步任务

1. 扩充标准动作模板和测试样本，完善实验数据集。
2. 将当前英文反馈提示改为中文提示，提升答辩展示效果。
3. 细化评分指标，尝试加入动作幅度、节奏一致性等多维评价。
4. 对不同动作类型分别设计评分策略，提高系统泛化能力。
5. 增加错误动作分类与更细粒度纠错建议。
6. 补充实验对比与性能分析，形成中期答辩所需的图表和结论。
7. 进一步优化界面或演示流程，提升系统展示完整度。
