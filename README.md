# 基于图像的健身动作评分系统

## 项目简介

本项目面向“基于图像的健身动作评分系统设计”课题，当前已完成一个基于人体姿态估计与 DTW（动态时间规整）的动作评分原型。系统输入一段标准动作视频和一段待测动作视频，先提取人体 33 个关键点，再对两段动作序列进行时序对齐，最终输出动作得分、局部偏差提示以及可视化结果。

当前实现重点放在“连续健身动作评分”这一主线流程上，适合用于中期答辩阶段展示算法设计、系统流程和阶段性实验结果。

## 目前已完成的工作

1. 完成了基于 MediaPipe Pose 的视频人体姿态关键点提取。
2. 实现了多人场景下的目标人物筛选，优先跟踪画面中心且时序连续的人体。
3. 实现了关键点归一化处理，降低人物位置、尺度差异对评分结果的影响。
4. 实现了姿态时序特征平滑与标准化，提升动作序列稳定性。
5. 实现了基于 DTW 的动作时序对齐，可处理动作速度不一致的问题。
6. 实现了 0-100 分的动作评分映射方法，可直观给出动作质量分数。
7. 实现了基于身体部位误差的反馈提示生成，可定位手臂、腿部、躯干等偏差。
8. 实现了结果导出功能，可输出 JSON 评分结果、对齐曲线图和叠加提示的视频。
9. 完成了俯卧撑、仰卧起坐等样例视频的测试与结果保存。

## 系统流程

1. 读取标准动作视频与待评分视频。
2. 使用 MediaPipe Pose 提取每一帧的人体 33 关键点。
3. 对关键点进行归一化、平滑和特征标准化处理。
4. 使用 DTW 对标准动作序列与待测动作序列进行时序对齐。
5. 根据归一化 DTW 距离计算最终得分。
6. 计算局部身体部位误差并生成动作纠正提示。
7. 输出 JSON、曲线图和可视化反馈视频。

## 主要文件说明

- `demo_dtw_scoring.py`：DTW 评分演示入口脚本。
- `fitness_action_eval/pose.py`：姿态提取、目标人物筛选与实时预览。
- `fitness_action_eval/dtw.py`：DTW 对齐与分数映射。
- `fitness_action_eval/feedback.py`：局部误差分析与提示生成。
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
python demo_dtw_scoring.py ^
  --ref_video input\pushup_test.mp4 ^
  --query_video input\pushup_wrong_test.mp4 ^
  --out_json output\dtw_coach_result.json ^
  --out_plot output\dtw_coach_plot.png ^
  --out_video output\dtw_coach_overlay.mp4
```

## 输入与输出说明

输入参数：

- `--ref_video`：标准动作参考视频。
- `--query_video`：待评分视频。
- `--out_json`：保存最终得分、DTW 距离和动作提示信息。
- `--out_plot`：保存 DTW 对齐可视化曲线图。
- `--out_video`：保存左右对比版评分视频。
- `--preview`：实时显示处理过程和最终叠加效果，按 `q` 可关闭窗口。

脚本内部固定配置：

- 姿态模型固定为 `pose_landmarker_lite.task`
- 单帧检测人数固定为 `1`
- 时间平滑窗口固定为 `7`
- 评分映射参数固定为 `6.0`
- 提示触发阈值固定为 `0.22`
- 提示最小间隔固定为 `8`
- 最大提示数量固定为 `40`

## 当前结果特点

1. 能够对连续动作视频进行整体评分。
2. 能够适应不同动作节奏下的序列对齐。
3. 能够输出局部偏差提示，而不仅仅是单一总分。
4. 已完成模块化拆分，便于后续扩展、调试与答辩展示。
5. 已支持左右双窗口对比预览与对比版结果视频导出。
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
