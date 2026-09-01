# Collab UAV+UGV — 空地协作闭环

UAV（无人机）× UGV（地面车）协同视觉语言导航：UAV 鸟瞰全局 + UGV 地面执行，根据目标照片协作到达目标点。

## 协作流程

```
┌─────────────────────────────────────────────────────────┐
│  1. UAV 启动改版 SPF（下视摄像头 → VLM 定位 → 飞行）     │
│  2. UGV 发送 CARLA 坐标给 UAV（共享内存）                │
│  3. UAV 在俯视图标注 CAR（蓝圈）和 GOAL（红圈）位置      │
│     并绘制方向线、距离标签 → 图片发给 UGV                │
│  4. UGV 根据标注鸟瞰图调用 VLM 规划路径 → 执行驾驶动作   │
│  5. UAV 独立通过改版 SPF 飞向目标                        │
│     任一 agent 到达目标 ≤10m → 成功                      │
└─────────────────────────────────────────────────────────┘
```

## 快速开始

### 终端 1 — 启动仿真

```bash
cd /home/zsn/VLN/carlaAir_experiments && conda activate carlaAir
./CarlaAir.sh
```

### 终端 2 — 生成任务场景

```bash
cd /home/zsn/VLN/carlaAir_experiments && conda activate carlaAir
python collab_UAV_UGV/scripts/setup_scene.py --episode-id town10hd_001
```

### 终端 3 — 开双视角（4画面：无人机前视/下视 + UGV FPV/Chase）

```bash
cd /home/zsn/VLN/carlaAir_experiments && conda activate carlaAir
python collab_UAV_UGV/scripts/show_dual_view.py
```

### 终端 4 — 运行空地协作

```bash
cd /home/zsn/VLN/carlaAir_experiments && conda activate carlaAir
python collab_UAV_UGV/scripts/run_collab.py --episode-id town10hd_001 --auto
```

## 批量自动测试

```bash
cd /home/zsn/VLN/carlaAir_experiments && conda activate carlaAir
bash collab_UAV_UGV/scripts/auto_run_collab.sh 5 town10hd_001
```

参数：`auto_run_collab.sh [运行次数] [episode_id]`

## 目录结构

```
collab_UAV_UGV/
├── README.md
├── episodes/ -> ../collab_task1/episodes/   # 共享 episode 数据
├── scripts/
│   ├── run_collab.py          # 空地协作主脚本（UAV + UGV 双线程）
│   ├── debug_ugv.py           # UGV 单独调试（无人机悬停标注 + 人工确认每步）
│   ├── setup_scene.py         # 场景初始化（生成车辆 + 定位无人机）
│   ├── show_dual_view.py      # 四画面实时查看（无人机 + UGV 视角）
│   └── auto_run_collab.sh     # 批量自动运行脚本
└── runs/                      # 实验输出（step图片、VLM响应、result.json）
```

## run_collab.py 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--episode-id` | Episode ID（必需），如 `town10hd_001` | — |
| `--time-limit` | 最大运行时间（秒） | 180 |
| `--auto` | 自动模式（无交互、自动保存日志） | False |

## 评估指标（result.json）

| 字段 | 说明 |
|------|------|
| `success` | 是否成功（任一 agent 到达目标） |
| `time_s` | 总耗时 |
| `uav_path_m` / `ugv_path_m` | UAV / UGV 轨迹长度 |
| `uav_final_dist_m` / `ugv_final_dist_m` | 最终距目标距离 |
| `uav_vlm_calls` / `ugv_vlm_calls` | VLM 调用次数 |
| `total_vlm_calls` | 总 VLM 调用次数 |

## 与 collab_task1 的关系

- `collab_task1/` — 单独 UAV / UGV 测试（run_spf_uav.py, run_spf_ugv.py）+ 原始协作脚本
- `collab_UAV_UGV/` — 独立的空地协作闭环，**不影响 collab_task1**
- 共享 `episodes/` 目录（通过符号链接）
