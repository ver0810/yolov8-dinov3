// Route Summary — Academic Blue / Seaborn Style
// 17-Class Fabric Defect Detection: 8 Exploration Routes → Innovation Reserve
#import "@preview/touying:0.6.1": *
#import themes.metropolis: *

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
)

// Seaborn muted palette
#let cBlue = rgb("#4C72B0")
#let cOrange = rgb("#DD8452")
#let cGreen = rgb("#55A868")
#let cRed = rgb("#C44E52")
#let cPurple = rgb("#8172B3")
#let cGrey = rgb("#8C8C8C")

#set text(font: ("DejaVu Sans", "Noto Sans SC"), size: 15pt)
#set table(stroke: 0.5pt + cGrey)

// ============================================================
// 1 封面
// ============================================================
#title-slide()

// ============================================================
// 2 路线总览
// ============================================================
#slide(title: [路线总览])[
  #grid(columns: (1fr, 1fr), gutter: 1.5em)[
    #text(fill: cBlue, weight: "bold")[已探索 8 条路线]
    #v(0.6em)
    1. 基线筛选 — YOLO 家族 7 模型，选定 YOLO11s
    2. 特征融合 — DINOv3 融合（全负）
    3. **注意力与损失** — PSA / 类别加权 / 过采样
    4. **分辨率杠杆** — 1280 训练-推理失配分析
    5. **数据增强池** — 8 池合成与采样（泄漏教训）
    6. **Transformer 突破** — RT-DETR（首个正收益 +13.5pp）
    7. **数据重划 & 云端** — V3 8:1:1 + Kaggle 双 T4
    8. **架构融合** — P2 辅助头（零推理开销）/ DEIMv2 调研
  ][
    #align(center)[
      #text(size: 11pt, fill: cGrey)[阶段性摸索 → 收敛]
      #v(0.3em)
      #table(
        columns: (auto, auto, auto),
        align: (left, center, center),
        [1 基线], [YOLO11s 最优], [#circle(fill: cGreen, radius: 3.5pt)],
        [2 DINOv3], [全负], [#circle(fill: cRed, radius: 3.5pt)],
        [3 注意力], [全负], [#circle(fill: cRed, radius: 3.5pt)],
        [4 分辨率], [1280 必要], [#circle(fill: cOrange, radius: 3.5pt)],
        [5 数据池], [混合], [#circle(fill: cOrange, radius: 3.5pt)],
        [6 RT-DETR], [*正收益*], [#circle(fill: cGreen, radius: 3.5pt)],
        [7 V3/Kaggle], [数据+算力基座], [#circle(fill: cBlue, radius: 3.5pt)],
        [8 融合], [P2 验证中], [#circle(stroke: cGrey + 1pt, fill: none, radius: 3.5pt)],
      )
      #v(0.5em)
      #align(center)[
        #grid(columns: 4, column-gutter: 1.5em, align: center + horizon,
          [#circle(fill: cGreen, radius: 3pt) #h(0.4em) #text(size: 9pt, fill: cGrey)[正]],
          [#circle(fill: cOrange, radius: 3pt) #h(0.4em) #text(size: 9pt, fill: cGrey)[混合]],
          [#circle(stroke: cGrey + 1pt, fill: none, radius: 3pt) #h(0.4em) #text(size: 9pt, fill: cGrey)[验证中]],
          [#circle(fill: cRed, radius: 3pt) #h(0.4em) #text(size: 9pt, fill: cGrey)[负]],
        )
      ]
    ]
  ]
]

// ============================================================
// 3 路线明细索引
// ============================================================
#slide(title: [路线明细索引])[
  #table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, left, center, center, left),
    [*No. 路线*], [*核心配置*], [*规模*], [*关键指标*], [*归档*],
    [1 基线], [YOLO11s \@640 100ep auto 优化], [7 模型], [mAP50 0.579→0.611], [`01_baseline`],
    [2 DINOv3], [convnext/vitms/hybrid 融合], [8 实验], [全负 0.05-0.32], [`035-043`],
    [3 注意力/损失], [PSA-p3/p4/p5 + 类权重 + 过采样], [12 实验], [全 ≤ 基线], [`009-017/030`],
    [4 分辨率], [640↔1280 交叉], [3 实验], [1280 必要，640 崩 0.31], [`018/019/022`],
    [5 数据池], [8 池 2985 合成 + 斑点], [4 实验], [泄漏教训 80.9%], [`061-075`],
    [6 RT-DETR], [RT-DETR-L \@1280 72ep AdamW], [3 变体], [*R\@0.25 0.7565*], [`095-097`],
    [7 V3/Kaggle], [V3 8:1:1 + T4×2], [15 实验], [079-093 体系], [`079-093`],
    [8 P2/DEIMv2], [P2 辅助头 + DEIM 调研], [2 变体], [验证中], [`098/099`],
  )
]

// ============================================================
// 4 Route 1 - 基线筛选
// ============================================================
#slide(title: [路线 1  ·  基线筛选])[
  #grid(columns: (1.2fr, 0.8fr), gutter: 1em)[
    #text(fill: cBlue, weight: "bold")[目标：7 模型家族中选定基线]
    - 训练：1445 图 / 验证 181 图 / 17 类 / 640 imgsz / 100ep
    - 指标：mAP50（`val conf=0.001` 全阈值积分）
    - 结论：*YOLO11s 最优*（mAP50 0.611），参数 9.43M
  ][
    #align(center)[
      #table(columns: (auto, auto, auto), align: (left, right, right),
        [*模型*], [*mAP50*], [*参数*],
        [YOLOv8n], [0.579], [3.01M],
        [YOLOv8s], [0.586], [11.14M],
        [YOLO11n], [0.587], [2.59M],
        [*YOLO11s*], [*0.611*], [*9.43M*],
        [YOLO26n], [0.557], [2.51M],
        [YOLO26s], [0.606], [9.96M],
        [World-v2s], [0.582], [12.76M],
      )
    ]
  ]
  #v(0.5em)
  #grid(columns: (1fr, 1fr), gutter: 1em)[
    #image("assets/class_distribution.png", width: 100%)
  ][
    #text(size: 11pt)[
      *家族 Δ*：YOLO11 +0.024（N→S），YOLO26 +0.049
      *效率*：Nano 0.9h / Small 1.0h \@3060
    ]
  ]
]

// ============================================================
// 5 Route 2 - DINOv3 融合（全负）
// ============================================================
#slide(title: [路线 2  ·  DINOv3 融合（全负）])[
  #grid(columns: (1fr, 1fr), gutter: 1.2em)[
    #text(fill: cRed, weight: "bold")[8 实验，R\@0.25 0.05-0.32，全负]
    - 融合方式：convnext / vitms / hybrid \*3 种
    - 原因：热力图证实“只在大目标有效”
    - 结论：不可逆，全负，关闭
  ][
    #table(columns: (auto, auto, auto), align: (left, right, right),
      [*实验*], [*R\@0.25*], [*mAP50*],
      [040 hybrid], [0.244], [0.32],
      [035 convnext], [0.10], [0.28],
      [036 vitms], [0.049], [0.19],
      [037 hybrid], [0.128], [0.31],
    )
    #text(size: 10pt, fill: cGrey)[vs 基线 0.614，差距 -0.3~-0.55]
  ]
  #v(0.4em)
  #align(center)[#text(fill: cGrey, size: 11pt)[对比：DINOv3 融合前/后热力图（占位） — 预留大图对比区]]
  #align(center)[#image("assets/dinov3_heatmap.png", width: 58%)]
]

// ============================================================
// 6 Route 3 - 注意力与损失
// ============================================================
#slide(title: [路线 3  ·  注意力与损失加权])[
  #grid(columns: (1fr, 1fr), gutter: 1.2em)[
    #text(fill: cRed, weight: "bold")[12 实验，全 ≤ 基线]
    #table(columns: (auto, auto, auto), align: (left, right, left),
      [*实验*], [*R\@0.25*], [*结论*],
      [009 PSA-p3], [0.497], [全负],
      [014 PSA], [0.586], [≈基线],
      [017 P2+SE], [0.498], [全负],
      [019 类权重], [0.621], [微增],
      [034 过采样], [0.543], [全负],
    )
  ][
    #text(fill: cBlue, weight: "bold")[损失补丁（环境变量门控）]
    - `SMALL_DEFECT_CLS` + `GAIN`（bd/hei 类加权）
    - `AREA`（小目标面积加权）
    - `NEG`（难例背景加权 `w=1+(NEG-1)p^γ`）
    #text(size: 10pt, fill: cGrey)[默认关闭，仅 019 等少数开启有效]
  ]
  #v(0.3em)
  #image("assets/route3_rat25_bar.png", width: 100%)
]

// ============================================================
// 7 Route 4 - 分辨率杠杆
// ============================================================
#slide(title: [路线 4  ·  分辨率杠杆])[
  #grid(columns: (1fr, 1fr), gutter: 1.2em)[
    #text(fill: cOrange, weight: "bold")[1280 必要，跨尺度失配崩溃]
    #table(columns: (auto, auto, auto), align: (left, right, right),
      [*模型*], [*R\@0.25 \@1280*], [*R\@0.25 \@640*],
      [004 \@640 训练], [0.309 崩], [0.614 基线],
      [019 \@1280 训练], [*0.627*], [0.608],
      [061 patch], [0.613], [0.593],
    )
    - 1280 训练 → 只能 1280 推理（1536/2048 失配）
    - 640 训练 → 640 推理
  ][
    #text(fill: cBlue, weight: "bold")[尺度失配量化]
    - 078：v2 放大把训练尺度拉至 0.078（11×）vs 推理 0.007
    - wv2 小波 0.020（2.8×）较近但仍偏
    - 结论：小目标中位 12px \@2048 → 7px \@1280
  ]
  #image("assets/route4_scale_hist.png", width: 100%)
]

// ============================================================
// 8 Route 5 - 数据增强池
// ============================================================
#slide(title: [路线 5  ·  数据增强池])[
  #grid(columns: (1fr, 1fr), gutter: 1.2em)[
    #text(fill: cOrange, weight: "bold")[8 池 + 3 教训]
    - 整图 1445 + bd/hei 中心 patch 583
    - 合成池 2388 + wavelet 2985
    - 078：R\@0.25 0.659（hei 0.462 提升）但 bd 0.182 无效
    - 076：R\@0.25 0.635（脚本宏 0.618）
  ][
    #text(fill: cRed, weight: "bold")[泄漏教训 · 80.9% 重叠]
    - 背景抑制 4 形态（065/066/067/070/071）全负
    - val 标注稀疏，压背景 = 压真阳性
    - 增强池必须无泄漏（源图 ∈ 训练集）
  ]
  #grid(columns: (1fr, 1fr), gutter: 1em)[
    #image("assets/class_distribution.png", width: 100%)
  ][
    #image("assets/route5_pool_pie.png", width: 100%)
  ]
]

// ============================================================
// 9 Route 6 - Transformer 突破
// ============================================================
#slide(title: [路线 6  ·  Transformer 突破（首个正收益）])[
  #grid(columns: (1.1fr, 0.9fr), gutter: 1em)[
    #text(fill: cGreen, weight: "bold")[RT-DETR-L \@1280：+13.5pp]
    #table(columns: (auto, auto, auto, auto), align: (left, right, right, right),
      [*模型*], [*R\@0.25*], [*P\@0.25*], [*Δ*],
      [019 YOLO11s], [0.621], [0.624], [—],
      [*096 RT-DETR-L*], [*0.757*], [0.403], [*+0.136*],
      [097 RT-DETR-L AdamW], [0.714], [0.394], [+0.093],
    )
    - 去 NMS 的 DETR 解码器，端到端匹配
    - 旧结论“所有训练侧无正收益”被推翻
  ][
    #align(center)[
      #image("assets/confusion_matrix_097_0548.png", width: 92%)
      #text(size: 9pt, fill: cGrey)[097 混淆矩阵（17 类，1280）]
    ]
  ]
  #v(0.3em)
  #grid(columns: (1fr, 1fr, 1fr), gutter: 0.6em)[
    #image("assets/tp_bd_pair.jpg", width: 100%)
    #image("assets/tp_heidian_pair.jpg", width: 100%)
    #image("assets/tp_big_pair.jpg", width: 100%)
  ]
  #align(center)[#text(size: 9pt, fill: cGrey)[左 GT / 右 097 预测（conf=0.548）— 小目标（bd 9.4px/hei 7.4px \@1280）检出案例]]
]

// ============================================================
// 10 Route 7 - 数据重划 & Kaggle
// ============================================================
#slide(title: [路线 7  ·  V3 重划 & Kaggle 云端])[
  #grid(columns: (1fr, 1fr), gutter: 1.2em)[
    #text(fill: cBlue, weight: "bold")[V3 8:1:1 + 类分层迭代]
    #table(columns: (auto, auto, auto), align: (left, center, center),
      [*集*], [*图像*], [*说明*],
      [train], [1448], [8 份],
      [val], [180], [1 份 每类 ≥8],
      [test], [179], [1 份],
    )
    - 泄漏检查：与 v1 val 重叠仅 7%
    - 小尺度池：524 图（131 图 ×4 原位变体）
  ][
    #text(fill: cBlue, weight: "bold")[Kaggle 双 T4×2]
    - `device 0,1` + batch 32 \@1280，100ep ≈3.5h
    - CLI `--dir-mode skip` 坑 + `data/data.yaml` 路径
    - 079：R\@0.25 0.584（bd 0.083 短板暴露）
    - 080：R\@0.25 0.604（small 池有效）
  ]
  #grid(columns: (1fr, 1fr), gutter: 1em)[
    #image("assets/class_distribution.png", width: 100%)
  ][
    #image("assets/route7_train_curve.png", width: 100%)
  ]
]

// ============================================================
// 11 Route 8 - 架构融合 & DEIMv2 调研
// ============================================================
#slide(title: [路线 8  ·  架构融合（P2 辅助头 / DEIMv2）])[
  #grid(columns: (1fr, 1fr), gutter: 1.2em)[
    #text(fill: cBlue, weight: "bold")[P2 辅助头（方案 A）]
    - 目标：bd/hei 漏检（FN 185 中 bd 31/44）
    - 设计：P2(stride4) 辅助头，推理零开销
    - 验证：前向 OK，103.8 GFLOPs（+0.3% 零开销）
    - 全接入 219 GFLOPs（8.5× 不可行）→ 已否决
    - 烟图：`rtdetr-l-p2aux.yaml` + `P2AuxHead`
  ][
    #text(fill: cPurple, weight: "bold")[DEIMv2 调研]
    - DEIM：Dense O2O + MAL，收敛快 50%
    - DEIMv2：DINOv3 骨干 + STA 多尺度
    - 风险：独立 engine（非 ultralytics），DINOv3 在织物域未验
    - 状态：DEIMv2 V1 烟图（HGNetv2-N 5ep 通，AP≈0 未收敛）
  ]
  #image("assets/route8_p2_arch.png", width: 100%)
]

// ============================================================
// 12 创新预留 - 1
// ============================================================
#slide(title: [创新实验预留  ·  一])[
  #rect(width: 100%, height: 75%, stroke: (dash: "dashed", paint: cBlue), radius: 6pt)[
    #align(center + horizon)[
      #text(fill: cBlue, size: 20pt, weight: "bold")[创新实验 · 待填空]
      #v(0.5em)
      #text(fill: cGrey, size: 13pt)[预留：最终创新方案的图表 / 消融 / 对比]
      #v(0.3em)
      #text(fill: cGrey, size: 11pt)[建议尺寸：16:9 横版大图 + 底部 3 列小图]
    ]
  ]
  #v(0.5em)
  #grid(columns: (1fr, 1fr, 1fr), gutter: 0.8em)[
    #rect(height: 18%, stroke: (dash: "dashed", paint: cGrey), radius: 4pt)[#align(center + horizon)[#text(fill: cGrey, size: 9pt)[对比图 1]]]
    #rect(height: 18%, stroke: (dash: "dashed", paint: cGrey), radius: 4pt)[#align(center + horizon)[#text(fill: cGrey, size: 9pt)[对比图 2]]]
    #rect(height: 18%, stroke: (dash: "dashed", paint: cGrey), radius: 4pt)[#align(center + horizon)[#text(fill: cGrey, size: 9pt)[对比图 3]]]
  ]
]

// ============================================================
// 13 创新预留 - 2
// ============================================================
#slide(title: [创新实验预留  ·  二])[
  #rect(width: 100%, height: 75%, stroke: (dash: "dashed", paint: cBlue), radius: 6pt)[
    #align(center + horizon)[
      #text(fill: cBlue, size: 20pt, weight: "bold")[创新实验 · 待填空]
      #v(0.5em)
      #text(fill: cGrey, size: 13pt)[预留：消融实验 / 逐类提升 / 失败案例]
      #v(0.3em)
      #text(fill: cGrey, size: 11pt)[建议：左侧大表 + 右侧 2×2 小图矩阵]
    ]
  ]
  #v(0.5em)
  #grid(columns: (1fr, 1fr), gutter: 0.8em)[
    #rect(height: 22%, stroke: (dash: "dashed", paint: cGrey), radius: 4pt)[#align(center + horizon)[#text(fill: cGrey, size: 9pt)[消融表]]]
    #rect(height: 22%, stroke: (dash: "dashed", paint: cGrey), radius: 4pt)[#align(center + horizon)[#text(fill: cGrey, size: 9pt)[逐类 R\@0.25 提升]]]
  ]
]

// ============================================================
// 14 结论
// ============================================================
#slide(title: [结论])[
  #grid(columns: (1fr, 1fr), gutter: 1.5em)[
    #text(fill: cBlue, weight: "bold")[阶段结论]
    - 最强模型：*096 RT-DETR-L\@1280*（R\@0.25 0.7565，+13.5pp）
    - 小目标：bd/hei 9.4/7.4px \@1280（\<40px cell），是主短板
    - 唯一数据正收益：CopyBlend 小目标原位池（080 +2.0pp）
    - 架构：P2 辅助头（零开销）+ DEIM 已具备可行性
  ][
    #text(fill: cRed, weight: "bold")[教训]
    - DINOv3 融合 / 注意力 / 长尾加权 / 背景抑制 全负
    - 增强池泄漏 80.9% 重叠教训
    - 报告口径：self-val + 0.25 + 1280（无编造）
  ]
  #v(0.8em)
  #align(center)[
    #text(fill: cBlue, weight: "bold", size: 16pt)[下一步：创新实验填空 → 正式训练 → 交付]
  ]
]
