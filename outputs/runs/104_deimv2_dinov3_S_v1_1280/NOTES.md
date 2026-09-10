# 104 DEIMv2 DINOv3-S V1 mixed @1280, RTX4090 24GB

- Pretrain: Intellindust/DEIMv2_DINOv3_S_COCO (COCO 80cls -> 17cls head remap)
- Config: configs/deimv2/v1_dinov3_S_1280_4090.yml — SAME recipe as 103 (L), single variable = backbone
  (official L recipe params: lr 5e-4 grouped, 68ep flatcosine, Dense O2O, batch8 x accum4 = 32, AMP,
   num_classes 17, warmup 1000 (scaled for batch8), intervitt distill backbone 192-dim)
- Data: V1 mixed 2838 train / v1 val 181 (same manifests as 102/103)

## Results (v1 val, conf=0.25 / IoU 0.5)
- MACRO R@0.25 = 0.7593  P@0.25 = 0.4753
- best macro-F1 = 0.6840 @ conf=0.51 (R 0.617 / P 0.768)
- COCO (native, best_stg2): AP50=0.696  AP50-95=0.498  (training final val)
- FPS @1280 fp16: 78.4ms on 3060 -> **12.8 FPS** (3080 折算 ~31.9 FPS — Item4 临界过线!)

## Comparison (same v1 val, conf=0.25)
| model | R@0.25 | P@0.25 | bestF1 | AP50 | FPS@3080 |
|---|---|---|---|---|---|
| 103 DINOv3-L | 0.7749 | 0.4598 | 0.7039@0.47 | 0.708 | 19.5 |
| **104 DINOv3-S** | **0.7593** | **0.4753** | **0.6840@0.51** | 0.696 | **31.9** |
| 096 RT-DETR-L | 0.7565 | 0.4029 | — | 0.676 | ~24 |

## Notes
- 训练中服务器暂停 2 次 + batch4->8 提速重启 1 次; resume 计数器重启导致一次 full-schedule rerun (权重从 ep27 续)
- best_stg1.pth(会话内 0.4967@ep63) / best_stg2.pth(final 0.4982) / best_stg1_prerestart.pth(保护快照) 三者归档
- 关键修复: vitt_distill.pth 需去除 `_model.` 前缀 (load_state_dict 严格模式)
- 归档: 2026-09-10T18:10+08:00
