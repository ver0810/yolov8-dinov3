# 102 DEIMv2 HGNetv2-L 32.2M V1 mixed 1280 @RTX4090 (vs 096 RT-DETR-L 32M)

- Source: remote 4090 24GB, /root/work/DEIMv2 outputs/v1_mixed_L_1280_4090
- Config: configs/deimv2/v1_mixed_L_1280_4090.yml (HGNetv2 B4, batch4 @1280, 72ep, lr 0.0001, val every 5ep)
- Data: V1 mixed 2838 train (b-final-v3 pool, 8136 anns) / 181 val, 17 classes
- Note: batch8 @1280 deformable attention OOM on 24GB -> batch4 + expandable_segments; server rebooted 2x mid-train, resumed from last.pth (checkpoint_freq 5)
- Weights: best_stg2.pth (epoch 71, final stage) -> weights/best_stg2.pth; last.pth also kept
- Final COCO eval (val 181): AP50=0.4106 AP50-95=0.5928 (EMA), best_stat AP@0.5 0.601 peak during training
- Training wall time: ~6h50m total (incl. 2 reboots), ~4.5min/ep @1280 batch4 on 4090
- Archiving time: 2026-09-09T15:10+08:00
