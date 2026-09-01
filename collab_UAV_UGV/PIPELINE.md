# 空地协作导航 Pipeline 梳理

> 对应代码：`carlaAir_experiments/collab_UAV_UGV/scripts/run_collab.py`
> 场景：CARLA-Air（CARLA + AirSim 单进程），Town10HD，目标为一辆静态 HGV 卡车。

本文档梳理无人机（UAV）+ 无人车（UGV）协作寻找静态目标的完整流程。

## 1. 总览

- **任务**：在时间预算（默认 180s）内，UAV 与 UGV **两者都**抵达目标卡车（平面距离 < 10m）。
- **架构**：两个并行线程 + 一块带锁的共享状态对象 `SharedState`，全部协作通过共享内存完成，无网络/消息队列。
  - **UAV 线程 `uav_loop`**：下视拍照 → 标注鸟瞰图（CAR/GOAL）→ 分享给 UGV；同时自己跑「改版 SPF」飞向目标。
  - **UGV 线程 `ugv_loop`**：上报自身位姿 → 等新鸟瞰图 → VLM 规划 10 点道路路径 → 闭环驾驶。

```
main()
 ├─ UAV 线程  uav_loop()   AirSim 无人机
 └─ UGV 线程  ugv_loop()   CARLA 无人车
        └── 二者只通过 SharedState 交换信息 ──┘
```

## 2. 入口 `main()`（L910）

1. `load_episode` 读 episode（注意：从 `town10hd_annotated.json` / `town10hd_templates.json` 读取）与模型配置 `load_config`。
2. `connect()` 连接 CARLA 与 AirSim，计算两坐标系偏置 `offset`。
3. `find_cars()` 找车：Mini Cooper = UGV 起点，HGV 卡车 = 目标。
4. 读无人机/车初始位姿，初始化 `SharedState`。
5. 起 UAV / UGV 两个 daemon 线程。
6. 主线程轮询 `stop_event` 或 `time_limit`（180s）。
7. 汇总结果：`ok = uav_success and ugv_success`（L1022），写 `result.json`。

## 3. 坐标系

| 系 | 使用者 | z 方向 |
|---|---|---|
| CARLA 世界系 | 无人车 | 向上 |
| AirSim NED 系 | 无人机 | 向下 |

- 偏置标定（L120）：`offset = [ap.x - dl.x, ap.y - dl.y, ap.z + dl.z]`（AirSim 位置 − CARLA 位置）。
- 转换（L124）：`carla_to_ned(loc) = offset + [x, y, -z]`。
- 像素 ↔ 世界（L211 `pix_to_world` / L227 `world_to_pix`）：平坦地面射线投射，FOV = 108°，假设地面平坦、相机垂直朝下。

## 4. 共享状态 `SharedState`（L37）

| 字段 | 含义 |
|---|---|
| `drone_pos` / `drone_yaw` | UAV 当前 NED 位姿 |
| `ugv_pos` | UGV 当前 CARLA 位姿 |
| `goal_ned` / `goal_carla` | 目标（真值）在 NED / CARLA 系坐标 |
| `annotated_image` + `image_timestamp` | UAV 标注好的鸟瞰图（UGV 读） |
| `annot_drone_pos` / `annot_drone_yaw` | **标注时刻**的无人机位姿（供 UGV 做像素→世界逆投影） |
| `car_px/py` / `goal_px/py` / `img_w/h` | CAR/GOAL 在鸟瞰图上的像素坐标与图片尺寸 |
| `stop_event` / `uav_success` / `ugv_success` | 停止信号与个体成功标志 |
| `uav_dc` / `ugv_dc` / `uav_path` / `ugv_path` | 指标（VLM 调用次数、轨迹） |

## 5. UAV 线程 `uav_loop`（L506）

每轮循环：

1. 读无人机 NED 位姿，算到 `goal_ned` 的距离 `dist`，写入 `state.drone_pos`。
2. **成功判定**：`dist < 10m` → 置 `uav_success=True`；但**不退出**，继续标注（还要喂给 UGV）。
3. 每 3s（`DECISION_INTERVAL`）一个决策步，做两件事：

   - **标注鸟瞰图**（L573–L620）：
     - 蓝圈 `CAR` = UGV 上报位姿 `ugv_pos` 投影到像素；
     - **红圈 `GOAL` = 目标真值 `goal_ned` 投影**（L584，注意：用真值，不是 VLM 定位点）；
     - 绿十字 = 无人机中心；黄线 = CAR→GOAL（参考线）；距离标签 `UAV->goal` / `CAR->goal`。
     - 把整张图 + 标注时刻无人机位姿（`annot_drone_pos/yaw`）+ CAR/GOAL 像素写进 `state`。
   - **改版 SPF 飞行**（L628–L659，仅 `not uav_done` 时执行）：
     - `vlm_uav` 在下视图里找红卡车 → 返回 `{point:[x,y], height:-1/0/1}`；
     - `pix_to_world` 把该像素转世界点 → `fly_horizontal` 水平飞 → `do_height` 按高度指令降/升/保持（下降有 5m 离地安全下限）。

> 当前顺序：**先标注（用真值 goal）→ 再 VLM 定位飞行**。VLM 的 `point` 只用于无人机自己飞，不参与 GOAL 标注。

## 6. UGV 线程 `ugv_loop`（L678）

每轮循环：

1. 读车 CARLA 位姿，算到 `goal_carla` 的距离，写入 `state.ugv_pos`、`state.goal_carla`。
2. **成功判定**：`dist < 10m` → 置 `ugv_success=True`，继续上报位姿。
3. 每 3s 一个决策步（仅 `not ugv_done`）：

   - **等新鸟瞰图**（L736，最多 10s）；等不到就「缓慢前爬」2s。
   - **VLM 规划**（L769）：`vlm_ugv` 吃鸟瞰图 + CAR/GOAL 像素 + 图片尺寸，输出 10 个像素路径点（首点=CAR、末点=GOAL、中间沿路）。
   - **合理性校验**（L785–L804）：路径长度 ≥2、首点不在 (0,0)/图外、首末点离 CAR/GOAL ≤200px，否则丢弃重试。
   - **像素 → CARLA 世界**（L809–L824）：用**标注时刻**的无人机位姿做逆投影，再减 `ned_to_carla_xy = goal_ned[:2] - goal_carla[:2]`（= 帧偏置）转回 CARLA 系。
   - **`follow_path` 闭环驾驶**（L411）：逐点追踪，航向误差离散为 6 种原语（前进/左转/右转/倒车/倒左/倒右）；卡死 >2s 倒车脱困；单点 15s sim 超时跳过；每过一个点检查是否到目标（硬编码 `< 10m`，L492）。
   - **停滞检测**（L843）：一轮下来位移 <1m → 自动倒车 3m。

## 7. 成功判定

- 个体成功：无人机 / 车各自 `dist < SUCCESS_DIST(=10m)` 即置 `*_success=True`。
- **最终成功（L1022）**：`ok = uav_success and ugv_success` —— **两者都到达才成功**。

## 8. 关键函数索引

| 函数 | 位置 | 作用 |
|---|---|---|
| `connect` | L99 | 连 CARLA/AirSim，算 `offset` |
| `pix_to_world` / `world_to_pix` | L211 / L227 | 像素 ↔ NED 世界（平坦地面） |
| `fly_horizontal` | L249 | 比例速度水平飞向目标 |
| `do_height` | L265 | 按高度指令下降/上升/保持 |
| `vlm_uav` | L285 | 下视图定位红卡车 → 像素 + 高度 |
| `vlm_ugv` | L323 | 鸟瞰图 → 10 点道路路径 |
| `_classify_action` | L395 | 航向误差 → 6 驾驶原语 |
| `follow_path` | L411 | 闭环路径追踪 + 卡死脱困 |
| `uav_loop` | L506 | UAV 主循环（标注 + SPF） |
| `ugv_loop` | L678 | UGV 主循环（规划 + 驾驶） |
| `save_result` | L876 | 写 `result.json` |

## 9. 已知不一致 / 注意事项

1. 模块 docstring 写 `either reaching the goal (≤10m) = success`（OR），但代码 L1022 实际是 **AND**。
2. `follow_path` 内成功阈值是**硬编码 `10.0`**（L492），未引用 `SUCCESS_DIST` 常量。
3. **GOAL 标注用的是目标真值 `goal_ned`**（L584），不是 VLM 定位点——即地图渲染环节依赖真值。
4. `DRONE_ALTITUDE = 60.0`（L28）已声明但流程中未引用（实际高度从 AirSim state 读取）。
5. `load_episode` 读取 `town10hd_annotated.json` / `town10hd_templates.json`。
