// 织物缺陷检测 · 大学生竞赛素材 PPT（基线 vs 最优方法）
// 干净简洁：仅文字 + 图/表，无动画无装饰
#import "@preview/touying:0.6.1": *
#import themes.metropolis: *

#show: metropolis-theme.with(aspect-ratio: "16-9")
#set text(font: ("DejaVu Sans", "Noto Sans SC"), size: 14pt)

// ============================================================
// 1 封面
// ============================================================
#title-slide(
  title: [织物缺陷检测：从数据难点到方法选择],
  subtitle: [17 类缺陷 · 2048×2048 原图 · RT-DETR 基线对比],
  extra: [大学生竞赛素材 · 2026-09],
)

// ============================================================
// 2 任务与数据集难点
// ============================================================
#slide[
  == 任务定义

  - 织物表面缺陷检测：17 类，框级定位（bbox）
  - 原图 2048×2048，训练 1445 / 验证 181 / 测试 181
  - 推理分辨率 1280（640/1280 训练模型不能混用）
  - 只提交训练权重，禁用 TTA / 集成等推理端手段

  #v(0.3em)
  #align(center)[#image("assets/class_distribution.png", width: 62%)]
]

// ============================================================
// 3 数据集难点
// ============================================================
#slide[
  == 数据难点（三条）

  #grid(columns: (1fr, 1fr), gutter: 1em)[
    #block[
      *难点 1：类别极不均衡 + 部分类样本极少*
      - bd/heidian 是"白点/黑点"，GT 仅 44/39 个实例
      - wy 污印 39 / zmty 37 / lj 12 / yy 压印仅 2 个
    ]
    #block[
      *难点 2：小目标尺度失配*
      - 中位 bd 15px、heidian 12px（2048）
      - 1280 推理下仅 9.4 / 7.4px，小于 40px 特征网格
      - mAP50 仅 0.15-0.29（小目标定位差）
    ]
    #block[
      *难点 3：跨类视觉相似 / 漏标注*
      - heidian→wy、zmty→bmss 等易混淆
      - val 标注稀疏：FP 高 ≠ 纯背景，压背景会压真阳性
    ]
    #image("assets/route4_scale_hist.png", width: 100%)
  ]
]

// ============================================================
// 4 数据处理方法
// ============================================================
#slide[
  == 数据处理（三步）

  #grid(columns: (1.25fr, 0.75fr), gutter: 1em)[
    [
      1. *尺寸对齐*：训练统一 1280，消除 640/1280 尺度失配崩溃
      2. *增强池扩充*：实例中心 patch（bd/hei 583）+ 合成池 + wavelet
      3. *无泄漏划分*：8:1:1 图级重划，类分层迭代（val 每类 ≥8）
      - 教训：跨集验证有 80.9% 泄漏，任何跨划分数值作废
      #v(0.3em)
      *注*：增强池提升小目标召回，但整体 mAP 收益有限——增强不解决尺度表征
    ]
    #align(center)[#image("assets/route5_pool_pie.png", width: 100%)]
  ]
]

// ============================================================
// 5 方法对比（基线 vs 最优）
// ============================================================
#slide[
  == 方法对比：YOLO 基线 vs RT-DETR 最优

  #grid(columns: (0.85fr, 1.15fr), gutter: 1em)[
    #table(
      columns: (auto, auto, auto),
      align: (left, right, right),
      inset: 4pt,
      [*方法*], [*R\@0.25*], [*mAP50*],
      [yolo11s 基线 004], [0.614], [0.605],
      [yolo11s+cls 019], [0.621], [0.577],
      [*RT-DETR-L 096*], [*0.757*], [0.677],
      [RT-DETR-L 097], [0.716], [0.636],
    )
    #align(center)[
      #image("assets/route3_rat25_bar.png", width: 100%)
      #text(size: 9pt)[R\@0.25 对比（v1 val）]
    ]
  ]
]
#slide[
  == RT-DETR 架构与创新点

  #align(center)[#image("assets/rtdetr_arch_official.png", width: 94%)]
  #align(center)[#text(size: 8pt, fill: gray)[图源：RT-DETR 官方架构图（arXiv:2304.08069，经 Ultralytics 文档引用）；P2 辅助头为本工作新增]]

  #grid(columns: (1.25fr, 0.75fr), gutter: 1em)[
    #text(size: 9pt)[*基线*：HGNetv2 P2–P5 → 混合编码器（AIFI + 跨尺度融合）→ IoU-aware query 选择 → 解码器（6 层×300 query，P3/P4/P5）→ 框 + 类别]
    #rect(stroke: rgb("#C44E52") + 1.2pt, radius: 4pt, inset: 6pt)[
      #text(size: 9pt)[*创新*：*P2AuxHead*（P2/4 轻量头，训练侧，推理零开销）]
    ]
  ]
]

// ============================================================
// 7 RT-DETR 训练配置与效果（数字）
// ============================================================
#slide[
  == RT-DETR 训练配置与效果

  #grid(columns: (1fr, 1fr), gutter: 1em)[
    #table(
      columns: (auto, auto),
      align: (left, left),
      inset: 4pt,
      [*项*], [*值*],
      [模型], [RT-DETR-L 32M, COCO 预训练],
      [数据], [b-final-v3（2838 池）],
      [训练], [1280, batch8, 72ep, AdamW 1e-4],
      [口径], [self-val + conf 0.25 + IoU 0.5],
      [R\@0.25], [*0.716*（vs YOLO 0.62）],
      [mAP50], [0.636],
      [FPS], [~30（3080 估，临界）],
    )
    #align(center)[
      #image("assets/route6_rtdetr_conf.png", width: 95%)
      #text(size: 9pt)[conf-R/P 曲线]
    ]
  ]
]

// ============================================================
// 8 小目标可视化（GT vs 预测）
// ============================================================
#slide[
  == 检出与漏检案例（左 GT / 右 预测）

  #grid(columns: (1fr, 1fr, 1fr), gutter: 0.5em)[
    #image("assets/tp_bd.jpg", width: 100%)
    #image("assets/tp_hei.jpg", width: 100%)
    #image("assets/tp_big.jpg", width: 100%)
  ]
  #align(center)[#text(size: 9pt)[bd / heidian 小目标检出 + 大目标检出（conf 0.55）]]

  #grid(columns: (1fr, 1fr, 1fr), gutter: 0.5em)[
    #image("assets/fn_bd.jpg", width: 100%)
    #image("assets/fn_hei.jpg", width: 100%)
    #image("assets/fp_bd.jpg", width: 100%)
  ]
  #align(center)[#text(size: 9pt)[漏检（bd/hei 7-20px）与误检——小目标仍是主短板]]
]

// ============================================================
// 8 结论
// ============================================================
#slide[
  == 结论

  1. 数据集核心难点：*类别不均衡 × 小目标尺度 × 类间相似*
  2. 数据处理：1280 对齐 + 增强池 + 无泄漏 8:1:1 重划
  3. 方法：*RT-DETR-L 最优*，R\@0.25 0.757（+13.5pp vs YOLO 基线），零跨类混淆
  4. 剩余短板：bd/hei 等极小目标（\<10px）召回仍不足——后续方向：小尺度表征 + 数据增强
]
