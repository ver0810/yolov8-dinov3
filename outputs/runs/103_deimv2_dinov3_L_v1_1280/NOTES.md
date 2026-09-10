# 103 DEIMv2 DINOv3-L (SOTA backbone) V1 mixed @1280, RTX4090 24GB

- Source: remote 4090, /root/work/DEIMv2 outputs/v1_dinov3_L_1280_4090
- Pretrain: Intellindust/DEIMv2_DINOv3_L_COCO (COCO 80cls) -> 17cls head remap (-t tuning path)
- Config: configs/deimv2/v1_dinov3_L_1280_4090.yml — official L recipe adapted @1280:
  lr 0.0005 grouped (dinov3 1.25e-5), AdamW wd 0.000125, 68ep flatcosine(flat34),
  Dense O2O aug policy [4,34,60], batch4 x grad_accum8 (=32 effective), AMP,
  num_classes 17, Normalize(mean/std), engine patched for grad_accum
- Data: V1 mixed 2838 train (b-final-v3 pool) / 181 val, 17 classes
- Weights: best_stg2.pth (ep66, EMA) + best_stg1.pth + last.pth (checkpoint00xx 快照未拉)
- Training: 7h05m total, ~6min/ep (incl. per-epoch val), 1 server reboot mid-run resumed

## Final results

### COCO eval (native, v1 val 181):
AP50=0.708  AP50-95=0.500  AP75=0.534
AP small=0.151 medium=0.328 large=0.563
AR@0.5=0.924  AR small=0.320

### Official scoring (conf=0.25, IoU=0.5, v1 val):
MACRO R@0.25 = 0.7749  P@0.25 = 0.4598  FP率 = 62.3%

vs 102 (HGNetv2-L from-scratch head): R 0.7060 -> 0.7749 (+6.9pp)
vs 096 (RT-DETR-L COCO-pretrained):  R 0.7565 -> 0.7749 (+1.8pp)  <- NEW BEST
P 0.4677 vs 0.4029(096) -> 0.4598 (better than 096, slightly below 102)

### Per-class R@0.25 (103):
wy 0.385(短) lj 0.417(短) zmty 0.540 heidian 0.590 bmss 0.591 bd 0.614
bj 0.773 zy 0.778 HD 0.778 cq 0.882 zj 0.889 cy 0.938 zyc/jt/yy/pd/jiaodai 1.000

### 短板分析:
- wy 0.385: 弱纹理混淆仍最差(但比 096 0.31 / 102 0.26 都好)
- lj 0.417: 新增短板(102 0.583/096 0.583), garbage 类对背景相似
- bd 0.614: 历史最高(019 0.217 -> 096 0.43 -> 103 0.614), DINOv3 表征对小缺陷有效
- 训练归档时间: 2026-09-10T09:35+08:00
