// Baseline Experiment Results — Phase 1
#import "@preview/touying:0.6.1": *
#import themes.metropolis: *

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
)

#set text(font: ("DejaVu Sans", "Noto Sans SC"), size: 16pt)

// ============================================================
// Slide 1: Title
// ============================================================

#title-slide(
  title: [工业缺陷检测基线实验],
  subtitle: [YOLOv8 / v11 / v26 / World-v2 Baseline Comparison],
  extra: [
    17-Class Fabric Defect Detection · Phase 1 Results \
    2026-07 · RTX 3060 6GB
  ],
)

// ============================================================
// Slide 2: Experiment Overview
// ============================================================

#slide[
  == 实验概览

  *数据集*
  - 训练：1,445 张 / 验证：181 张 / 测试：181 张
  - 17 个缺陷类别，图像尺寸 2448 × 2048
  - 数据增强：mosaic, hsv, flip 等默认配置

  #v(0.5em)
  *模型家族* (共 7 组实验)

  #table(
    columns: (auto, auto, auto, auto, auto),
    stroke: 0.5pt,
    align: center,
    inset: 5pt,
    [*No.*], [*Model*], [*Size*], [*Params*], [*Pretrained*],
    [001], [YOLOv8], [Nano], [3.01M], [COCO],
    [002], [YOLOv8], [Small], [11.14M], [COCO],
    [003], [YOLO11], [Nano], [2.59M], [COCO],
    [004], [YOLO11], [Small], [9.43M], [COCO],
    [005], [YOLO26], [Nano], [2.51M], [COCO],
    [006], [YOLO26], [Small], [9.96M], [COCO],
    [007], [YOLOWorld], [Small v2], [12.76M], [CLIP+COCO],
  )

  #v(0.5em)
  *训练参数* (默认/推荐参数)
  - epochs: 100 · imgsz: 640 · batch: 16 · lr0: 0.01 · patience: 100 · optimizer: auto
]

// ============================================================
// Slide 3: Results Table
// ============================================================

#slide[
  == 实验结果总览

  #set text(size: 11pt)

  #let h(fill, body) = table.cell(fill: fill, [#body])

  #table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    stroke: 0.5pt,
    align: center,
    inset: 4pt,
    [*No.*], [*Model*], [*mAP50↑*], [*mAP50-95↑*], [*Precision*], [*Recall*], [*Time*], [*Size*],

    [001], [yolov8n], [0.5793], [0.3970], [0.6119], [0.5656], [0.9h], [6.3MB],
    [002], [yolov8s], [0.5863], [0.3808], [0.6605], [0.5483], [1.1h], [22.5MB],
    [003], [yolo11n], [0.5872], [0.3954], [0.6783], [0.5614], [0.9h], [5.5MB],

    h(rgb("#DBE8F5"))[004], h(rgb("#DBE8F5"))[*yolo11s*], h(rgb("#DBE8F5"))[*0.6112*], h(rgb("#DBE8F5"))[*0.4123*], h(rgb("#DBE8F5"))[0.7765], h(rgb("#DBE8F5"))[0.5350], h(rgb("#DBE8F5"))[1.0h], h(rgb("#DBE8F5"))[19.2MB],

    [005], [yolo26n], [0.5568], [0.3853], [0.5947], [0.4951], [0.9h], [5.4MB],
    [006], [yolo26s], [0.6062], [0.4040], [0.5470], [0.6037], [1.0h], [20.3MB],
    [007], [worldv2], [0.5816], [0.3997], [0.6113], [0.5397], [1.0h], [25.8MB],
  )

  #v(0.5em)
  #text(size: 10pt)[All metrics from best mAP50 checkpoint. Green row = best overall.]
]

// ============================================================
// Slide 4: Model Family Comparison
// ============================================================

#slide[
  == 模型家族对比 (mAP50)

  #set text(size: 13pt)

  #table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    align: center,
    inset: 6pt,
    [*Family*], [*Nano mAP50*], [*Small mAP50*], [*Δ Diff*],
    [YOLOv8], [0.5793], [0.5863], [+0.0070],
    [YOLO11], [0.5872], [*0.6112*], [*+0.0240*],
    [YOLO26], [0.5568], [0.6062], [+0.0494],
    [YOLOWorld], [---], [0.5816], [---],
  )

  #v(1em)
  *关键发现*
  - YOLO11-Small 最佳: mAP50 = 0.6112, mAP50-95 = 0.4123
  - YOLO26-Small 第二: mAP50 = 0.6062，Nano→Small 提升最大 (+0.0494)
  - YOLO11-Nano (0.5872) 超越 YOLOv8-Small (0.5863)
  - YOLOWorld-v2: 封闭集微调后接近标准检测模型
]

// ============================================================
// Slide 5: Detailed Analysis
// ============================================================

#slide[
  == 详细分析

  *Nano 模型对比*
  - YOLO11n 最优 (mAP50 0.5872)，参数最少 (2.59M)
  - YOLOv8n 第二 (mAP50 0.5793)，参数 3.01M
  - YOLO26n 最弱 (mAP50 0.5568)，参数 2.51M

  #v(0.5em)
  *Small 模型对比*
  - YOLO11s 领先 (mAP50 0.6112)，YOLO26s 第二 (0.6062)
  - YOLOv8s 第三 (0.5863)，YOLOWorld-v2s 中等 (0.5816)

  #v(0.5em)
  *Precision-Recall (best mAP50 epoch)*
  - YOLO11s: 高 P (0.777) 低 R (0.535) → 漏检较多
  - YOLO26s: 高 R (0.604) 低 P (0.547) → 误检较多
  - YOLO11n: P/R 最均衡 (0.678 / 0.561)

  #v(0.5em)
  *效率* — 所有 Nano 训练约 0.9h，Small 约 1.0h
]

// ============================================================
// Slide 6: Inference Samples (YOLO11s) — 1/4
// ============================================================

#slide[
  == 推理可视化 — YOLO11s (1/4)

  #set text(size: 6pt)

  #grid(
    columns: (1fr, 1fr),
    rows: (auto, 1fr, auto, 1fr),
    gutter: 0.3em,
    align(center)[*bqc 不良情况* · conf 0.25],
    align(center)[*cf 成分* · 误检],
    image("../../outputs/inference_samples/bqc_12__0812.png", width: 100%),
    image("../../outputs/inference_samples/cf_3461__0724.png", width: 100%),
    align(center)[*chy 差异* · conf 0.30],
    align(center)[*dmg 损坏* · conf 0.53],
    image("../../outputs/inference_samples/chy_1465_XBW_20250213.png", width: 100%),
    image("../../outputs/inference_samples/dmg_133__XBW_20241116.png", width: 100%),
  )
]

// ============================================================
// Slide 7: Inference Samples — 2/4
// ============================================================

#slide[
  == 推理可视化 — YOLO11s (2/4)

  #set text(size: 6pt)

  #grid(
    columns: (1fr, 1fr),
    rows: (auto, 1fr, auto, 1fr),
    gutter: 0.3em,
    align(center)[*hs 灰色* · conf 0.45],
    align(center)[*hw 花纹* · conf 0.63],
    image("../../outputs/inference_samples/hs_1106__0921.png", width: 100%),
    image("../../outputs/inference_samples/hw_1086__XBW_20241116.png", width: 100%),
    align(center)[*lj 聚集* · conf 0.72],
    align(center)[*mh 模糊* · conf 0.40],
    image("../../outputs/inference_samples/lj_1126_DHW_20250103.png", width: 100%),
    image("../../outputs/inference_samples/mh_2338_DHW_20250103before.png", width: 100%),
  )
]

// ============================================================
// Slide 8: Inference Samples — 3/4
// ============================================================

#slide[
  == 推理可视化 — YOLO11s (3/4)

  #set text(size: 6pt)

  #grid(
    columns: (1fr, 1fr),
    rows: (auto, 1fr, auto, 1fr),
    gutter: 0.3em,
    align(center)[*sh 水痕* · conf 0.73],
    align(center)[*sw 缩水* · conf 0.64],
    image("../../outputs/inference_samples/sh_1403_DHW_20250103before.png", width: 100%),
    image("../../outputs/inference_samples/sw_1821_LZW_20250208.png", width: 100%),
    align(center)[*tss 脱色* · conf 0.75],
    align(center)[*xw 纤维* · conf 0.83],
    image("../../outputs/inference_samples/tss_LZW_20260124_152.png", width: 100%),
    image("../../outputs/inference_samples/xw_1045_DHW_20250103.png", width: 100%),
  )
]

// ============================================================
// Slide 9: Inference Samples — 4/4
// ============================================================

#slide[
  == 推理可视化 — YOLO11s (4/4)

  #set text(size: 6pt)

  #grid(
    columns: (1fr, 1fr),
    rows: (auto, 1fr, auto, 1fr),
    gutter: 0.3em,
    align(center)[*yq 油污* · 漏检],
    align(center)[*zf 折痕* · conf 0.34],
    image("../../outputs/inference_samples/yq_1099__20241017.png", width: 100%),
    image("../../outputs/inference_samples/zf_1721_DHW_20250103before.png", width: 100%),
    align(center)[*zmty 正面套印* · conf 0.81],
    align(center)[*zy 折印* · 漏检],
    image("../../outputs/inference_samples/zmty_1199_DHW_20250103.png", width: 100%),
    image("../../outputs/inference_samples/zy_274__0812.png", width: 100%),
  )
]

// ============================================================
// Slide 10: Summary & Next Steps
// ============================================================

#slide[
  == 总结与下一步

  *本次实验结论*
  - YOLO11s 在 17 类缺陷检测中表现最佳 (mAP50 0.6112)
  - YOLO11 系列整体优于 YOLOv8 和 YOLO26
  - YOLOWorld-v2 封闭集微调不劣于标准检测模型
  - Nano 中 YOLO11n 以最小参数量 (2.59M) 获最优效果
  - YOLOE 仅发布 seg 预训练权重，无法用于检测

  #v(1em)
  *后续方向*
  - DINOv3 backbone 融合实验
  - YOLOE 检测预训练权重（待发布）
  - Class-wise error analysis
  - Test set final evaluation
]

// ============================================================
// Slide 11: Appendix — Training Parameters
// ============================================================

#slide[
  == 附录: 训练参数详情

  #set text(size: 12pt)

  #table(
    columns: (auto, auto),
    stroke: 0.5pt,
    inset: 5pt,
    align: (left, left),
    [*Parameter*], [*Value*],
    [epochs], [100],
    [imgsz], [640],
    [batch], [16],
    [lr0], [0.01],
    [lrf], [0.01],
    [momentum], [0.937],
    [weight_decay], [0.0005],
    [warmup_epochs], [3.0],
    [optimizer], [auto],
    [patience], [100 (no early stop)],
    [mosaic], [1.0],
    [mixup], [0.0],
    [close_mosaic], [10],
    [amp], [True],
    [device], [NVIDIA RTX 3060 6GB],
    [dataset], [1,445 train / 181 val / 181 test],
    [nc], [17 classes],
  )
]
