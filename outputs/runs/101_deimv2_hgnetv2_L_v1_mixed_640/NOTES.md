# 101 DEIMv2 HGNetv2-L 32.2M V1 mixed 640 (vs RT-DETR-L 32M, N 3.55M)

- Source: third_party/DEIMv2/outputs/v1_mixed_L_640
- Config: configs/deimv2/v1_mixed_L_640.yml (HGNetv2 B4, 256 hidden, 6 layers, 640, batch4, 20ep)
- Data: V1 mixed 2838 train (b-final-v3 pool, 8136 anns) / 181 val, 17 classes
- Note: L @1280 混合池首 iter nan (D-FINE 分布回归对 <1px 极小框敏感), 640 规避
- Weights: best_stg1.pth (epoch 19) -> best.pt (AP50 0.474)
- Archiving time: 2026-09-07T03:00:58+08:00
