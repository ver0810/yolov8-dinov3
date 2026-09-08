---
title: YOLO-Master 实验归档（LoRA/PEFT 路线探索）
date: 2026-08-13
author: 实验验证组
status: 已终止并清理（仓库/venv/权重/脚本全部移除）
purpose: 记录 PEFT/LoRA 路线的探索过程、速度结论与放弃原因，避免未来重复踩坑
motivation: 论文 YOLO-PEFT（arXiv 2608.07051，腾讯+厦大）显示 planner-LoRA 在 VOC 上优于
  Full-SFT（yolo11s +7.1 mAP50-95、显存 -44%）；想验证"冻结主干+低秩适配"能否避免
  全量微调对弱类（bd/heidian）的破坏（065–067 loss 加权实验的教训）
related:
  - presentations/per_class_map.md
  - 实验 065/066/067（背景抑制，已证伪）
---

# YOLO-Master 实验归档（2026-08-13，已终止）

## 时间线与动作

1. clone 腾讯 YOLO-Master（ultralytics 8.4.101 fork，含 LoRA/MoLoRA/MoE/planner）到 `third_party/YOLO-Master`（自带 .venv：ultralytics 8.4.101 + peft + torch 2.5.1 cu124）
2. 4 个并行 scout 研究（core src / tests / configs / scripts-docs）→ 产出该仓库的 AGENTS.md（已随仓库删除）
3. **LoRA smoke 通过**：`apply_lora(r=16)` → 可训练 1.63M（15.9%）、适配器 780K（7.6%）、base 冻结；`save_lora_only/load_lora/merge_lora` 全生命周期 OK（merge 后回到标准模型，推理零变化——合规"只交权重"）
4. **PEFT Planner**：yolo11s → ADAPT（budget 2.1M，cold-start 先验）；EsMoE-N/S → ADAPT（建议 r=8 + include_attention）
5. EsMoE-N/S 权重下载（release，HTTP 200）、加载正常（2.69M / 9.71M 参数）

## 速度结论（核心教训）

| 配置 | 实测速度 | 原因 |
|---|---|---|
| yolo11s + LoRA（peft 后端）1280/b8 | **7s/it**（181 iters ≈ 21min/ep，60ep ≈ 21h） | 54 层全部 backbone conv 的 unfold 中间张量爆炸：P2 层 @1280 单层 unfold ~7.6GB（batch8）；单层 fwd+bwd 115ms × 54 层 ≈ 6.2s/it |
| yolo11s + LoRA 640/b16 | 预估 1.5–2s/it | unfold 张量 1/4 |
| EsMoE-N 640/b16 全量 | **8.7s/it**（smoke 782s/90 iters） | MoE 稀疏路由（gather/scatter）反向开销；推理仅 11ms/图 |
| torch.compile | **崩溃**（terminate called，WSL2 + dynamo 不兼容 MoE 图） | — |

其他发现：
- **peft 版本坑**：requirements pin `peft>=0.18.0,<0.20.0`，venv 装的是 0.20.0（超上限）→ 降到 0.19.1；缺 pandas/seaborn 一并补装
- **conv-native 快速路径已验证**（ΔW = B·Aᵀ reshape 成卷积核 + F.conv2d，与 unfold 数值等价 diff=0.0，单层 83ms vs 115ms）——用户要求撤销，未用于训练；若未来再碰 LoRA 高分辨率训练，这是已知的提速+省显存改造点
- 层过滤参数（LoRAConfig.from_layer/to_layer）：跳过 P2/P3 高分辨率层可进一步减 unfold（只适配 P4/P5/neck/head）

## 放弃原因（决策记录）

1. **不解决核心瓶颈**：bd/heidian 的诊断问题是"信号弱（45% GT 无响应）+ 尺度（P3 <1px）"——MoE 是特征变换器，不增加特征分辨率、不改变尺度失配；top-2 稀疏路由对弱特征（微小缺陷）不稳定
2. **训练成本扼杀迭代**：MoE 8.7s/it → 100ep 22h；LoRA 1280 21h——试错空间为零，效果不确定性高
3. **容量/历史参照**：EsMoE-N 仅 2.68M（yolo11n 参照 003=0.5803）；架构换血历史全败（DINOv3/yolo26s/P2）
4. 预估 068（EsMoE-N@640）宏 R@0.25 仅 0.50–0.58，不达 004 基线（0.6140）概率高

## 清理记录

- `third_party/YOLO-Master/`（含 .venv）、`.venv-ym`、`scripts/ym_train_lora.py`、`ym_lora_smoke.py`、`EsMoE-N/S.pt`、`outputs/runs/068_esmoe_n_640` 全部删除
- 主项目 AGENTS.md 中 YOLO-Master 段落已移除；主 venv（ultralytics 8.4.104）未受影响

## 未来若重启该路线的要点（备忘）

1. LoRA 高分辨率（1280）必须：conv-native 实现（避免 unfold）+ 层过滤（from_layer=7）
2. MoE 训练成本：先 smoke 1ep 测速再决定投入；torch.compile 在 WSL2 不可用
3. planner 审计 JSON 在 `runs/planner_audit/`（已随仓库删除）
4. peft 需满足 fork 的版本 pin（0.19.x），否则行为不可预期
