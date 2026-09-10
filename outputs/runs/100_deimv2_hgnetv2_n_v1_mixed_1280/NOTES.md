# 100 DEIMv2 HGNetv2-N V1 mixed 1280 (vs 099 V1 1280)

- Source: third_party/DEIMv2/outputs/v1_mixed_1280
- Config: configs/deimv2/v1_mixed_1280.yml (HGNetv2 B0, 128 hidden, 3 decoder layers, 1280, batch4, 20ep, Dense O2O disabled)
- Data: V1 mixed 2838 train (b-final-v3 pool, 8136 anns) / 181 val (377 anns), 17 classes
- Compare: 099 V1 1280 (1445 train, 3042 anns) same 20ep, same model, same 1280
- Weights: best_stg1.pth -> best.pt, last.pth
- Archiving time: 2026-09-03T20:30:09+08:00
