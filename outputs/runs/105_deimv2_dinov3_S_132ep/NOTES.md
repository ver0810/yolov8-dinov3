# 105 DEIMv2 DINOv3-S 官方 132ep 配方 HyperOpt（vs 104 68ep）

- Pretrain: Intellindust/DEIMv2_DINOv3_S_COCO（同 104）→ 17cls 头重映射
- Config: configs/deimv2/v1_dinov3_S_132ep_official.yml —— 官方 S 配方全套：
  epoches 132, flat_epoch 64, no_aug 12, 增强锚点 [4,64,120], stop 120,
  batch8 x grad_accum4 (=32), warmup 1000（等比缩放）, num_classes 17, Normalize, val/5ep gate
- Data: 与 102/103/104 完全相同清单（v1 mixed 2838 / v1 val 181）

## 训练事故记录（不影响结论可信度）
1. output_dir 漏改 → 105 输出写入 104 目录（best_stg1/last 被覆盖）——本地 104 权重归档完好，无损失
2. 平台暂停杀训练 1 次（02:39，ep81 处）
3. 续跑两次踩 pre-loop val 未定义 epoch（已 revert 该块，loop 内 5ep gate 保留）
4. 105 产物已用 /root/work/105_guard/ 隔离快照

## 结果（COCO AP50-95, v1 val 181 图）
- **105 best = 0.4933 @ep84**（best_stg1.pth）；final val 0.489（ep128）
- 104 best = **0.4982**（best_stg2, ep67）
- val 曲线：共同区间（ep0-58）两条几乎重合；104 cosine 尾段 +0.6pp 收敛 0.498；
  105 flat64 段无增益，cosine 尾段（ep98-128）0.489-0.493 未超 104

## HyperOpt 结论
- **132ep 没有超越 68ep**——"S 欠训练"假设不成立
- 68ep cosine 已达 S 官方配方上限；官方 flat64 长调度对 2838 图小数据无益
- 与 102→103 的教训一致：**配方对齐已到位时，时长不是杠杆**
- 超越 103 的路径确认收敛到数据侧（wy/zmty 定向池 + 标注策略）

## 归档
- 105_guard_snapshot/（前段 ep0-80 的 best_stg1/stg2/last）
- best_stg1.pth（会话内 best 0.4933）+ last.pth（ep131）
- train105_initial.log + train105_resume.log（两段完整日志）
- 归档时间: 2026-09-11T11:10+08:00
