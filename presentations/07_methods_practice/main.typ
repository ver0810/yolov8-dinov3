// 四条技术路线的完整实践 — 竞赛素材 PPT v2
// 重图表叙述：每条方法 3-4 页（架构/训练/结果/可视化）
#import "@preview/touying:0.6.1": *
#import themes.metropolis: *

#show: metropolis-theme.with(aspect-ratio: "16-9")
#set text(font: ("DejaVu Sans", "Noto Sans SC"), size: 14pt)

#let cBlue = rgb("#4C72B0")
#let cOrange = rgb("#DD8452")
#let cGreen = rgb("#55A868")
#let cRed = rgb("#C44E52")
#let cGrey = rgb("#8C8C8C")

// ============================================================
// 1 封面
// ============================================================
#title-slide(
  title: [四条技术路线的完整实践],
  subtitle: [织物缺陷检测 17 类 · 从 YOLO 基线到 DINOv3 · R\@0.25 +31%],
  extra: [81 个训练 run · 6 条数据管线 · 3 套评测协议 · 2026-09],
)

// ============================================================
// 2 四方法总览
// ============================================================
#slide[
  == 四方法总览

  #table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, left),
    inset: 4pt,
    [*路线*], [*代表 run*], [*参数量*], [*R\@0.25*], [*FPS\@3080*], [*定位*],
    [① YOLO 家族], [004 / 019], [9.4M], [0.614 / 0.621], [~62], [轻量基线·速度充裕],
    [② RT-DETR-L], [096], [32M], [0.757], [~24], [Transformer 突破·速度临界],
    [③ DEIMv2-L], [103], [32.2M], [*0.775*], [19.5], [精度上限·速度不足],
    [④ DEIMv2-S], [104], [9.7M], [0.759], [*31.9*], [*双达标*（精度近上限+速度过线）],
  )

  #v(0.3em)
  #align(center)[#image("assets/fps_accuracy_scatter.png", width: 88%)]
]

// ============================================================
// 3 方法一：YOLO 家族（选型）
// ============================================================
#slide[
  == ① YOLO 家族：基线筛选（58 个 run）

  #grid(columns: (1.2fr, 0.8fr), gutter: 1em)[
    [
      - 系统扫过 *yolov8n/s、yolo11n/s/m、yolo26n/s、yoloworld_s* 七个模型
      - 选定 *yolo11s*（004）：参数/精度/速度最佳折中，R\@0.25 *0.614*
      - 全部在 v1 val、conf=0.25 同口径下对比——48 个变体逐一验证
    ]
    #align(center)[#image("assets/class_distribution.png", width: 100%)]
  ]
]

// ============================================================
// 4 方法一：结构改进（全负）与正收益项
// ============================================================
#slide[
  == ① YOLO 家族：改进尝试与两个正收益

  #align(center)[#image("assets/yolo_family_sweep.png", width: 96%)]

  #grid(columns: (1fr, 1fr), gutter: 1em)[
    #text(size: 9pt)[
      *正收益（绿）*：
      - 004 基线本身：P2 特征图保留 + 默认增强合适
      - 019 类别加权（SMALL_DEFECT_CLS gain=6）：+0.7pp，bd/heidian 专用
    ]
    #text(size: 9pt)[
      *负结果（蓝，60 个变体）*：
      - PSA/C2PSA/SE 注意力 ×6：全部 -0.03~-0.18
      - strong-aug / 300ep / copypaste / oversample / DINOv3-fusion ×8：全无增益
    ]
  ]
]

// ============================================================
// 5 方法一：训练侧工具箱（loss 补丁）
// ============================================================
#slide[
  == ① YOLO 家族：训练侧 loss 补丁（自研）

  #align(center)[#image("assets/route4_scale_hist.png", width: 74%)]

  #text(size: 9pt)[
    诊断：bd 中位 12px / heidian 11px，1280 训练时仅 7-9px——*尺度失配是主短板*。
    为此打了环境变量门控的 loss 补丁（v8DetectionLoss）：类级 cls 权重、面积感知加权、难例背景加权。
    *可复现*：`SMALL_DEFECT_CLS/GAIN/AREA/NEG` 四个开关，全部记录在 fork 的 loss.py 中。
  ]
]

// ============================================================
// 6 方法二：RT-DETR 引入
// ============================================================
#slide[
  == ② RT-DETR：Transformer 首次突破（+13.5pp）

  #align(center)[#image("assets/rtdetr_arch_official.png", width: 92%)]
  #align(center)[#text(size: 8pt, fill: gray)[图源：RT-DETR 官方架构（arXiv:2304.08069）]]

  #table(
    columns: (auto, auto, auto),
    align: (left, right, right),
    inset: 4pt,
    [*run*], [*配置*], [*R\@0.25*],
    [095], [\@640, 单卡 72ep], [0.707],
    [*096*], [*\@1280, 双 T4 72ep*], [*0.757*],
    [097], [AdamW 显式 + 1280], [0.716],
  )
  #align(center)[#text(size: 9pt)[*结论*：transformer + 原生 1280 训练 = 当时最强；640→1280 分辨率是最大杠杆]]
]

// ============================================================
// 7 方法二：RT-DETR 的速度困境
// ============================================================
#slide[
  == ② RT-DETR：精度可过、速度临界

  #grid(columns: (1fr, 1fr), gutter: 1em)[
    #table(
      columns: (auto, auto),
      align: (left, left),
      inset: 4pt,
      [*项*], [*值*],
      [参数/计算量], [32M / ≈376 GFLOPs\@1280],
      [FPS \@3080 折算], [~24（PyTorch fp16）],
      [Item4 ≥30 FPS], [*未过线*],
      [TensorRT 路径], [估 ~35-40，临界过],
      [交付约束], [仅交训练权重（TRT engine 存疑）],
    )
    #align(center)[
      #image("assets/fps_accuracy_scatter.png", width: 100%)
      #text(size: 9pt)[四代模型的速度-精度全景]
    ]
  ]
]

// ============================================================
// 8 方法三：DEIMv2 引入（为什么是它）
// ============================================================
#slide[
  == ③ DEIMv2：把 DINOv3 表征带进实时检测

  #align(center)[#image("assets/deimv2_sta_arch.png", width: 90%)]
  #align(center)[#text(size: 8pt, fill: gray)[图源：DEIMv2 官方论文 arXiv:2509.20787]]

  #grid(columns: (1.15fr, 0.85fr), gutter: 1em)[
    #text(size: 9pt)[
      *选型逻辑*：
      - 织物短板 = *弱纹理细粒度判别*（wy↔zmty↔heidian）——自监督 DINOv3 表征的强项
      - STA 适配器把 ViT 单尺度输出转 1/8、1/16、1/32 多尺度
      - 与 RT-DETR-L 同 32M 档，公平对照
    ]
    #rect(stroke: cGreen + 1.2pt, radius: 4pt, inset: 6pt)[
      #text(size: 9pt)[*三个官方创新点*：
        STA 空间调谐适配器 / 简化解码器（SwiGLU+RMSNorm+共享位置）/ 升级 Dense O2O（Copy-Blend）]
    ]
  ]
]

// ============================================================
// 9 方法三：配方踩坑（102）→ 修复（103）
// ============================================================
#slide[
  == ③ DEIMv2-L：配方决定成败（102 → 103）

  #table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    inset: 4pt,
    [*项*], [*102 首次（错）*], [*103 修正（官方）*],
    [lr], [1e-4 保守值], [*5e-4 分层（差 5×）*],
    [分类头], [80 类漏配], [*17 类重映射*],
    [batch/梯度累积], [4，无累积], [*4×8=32*],
    [增强], [无 Mosaic/MixUp], [*官方 Dense O2O*],
    [归一化], [缺 Normalize], [*ImageNet mean/std*],
  )

  #grid(columns: (1fr, 1fr), gutter: 1em)[
    #align(center)[#image("assets/v103_pr_curve.png", width: 100%)]
    #text(size: 9pt)[
      *同数据同架构，只改配方*：
      AP50 0.594 → *0.708*（+11.4pp）
      R\@0.25 0.706 → *0.775*（+6.9pp）
      best-F1 0.614 → *0.704*

      #v(0.2em)
      *方法论*：SOTA 架构必须配官方训练配方——直接沿用保守参数会浪费 7pp
    ]
  ]
]

// ============================================================
// 10 方法三：103 逐类效果
// ============================================================
#slide[
  == ③ DEIMv2-L：逐类结果与短板归因

  #grid(columns: (1.15fr, 0.85fr), gutter: 1em)[
    #align(center)[#image("assets/v103_item1_gap.png", width: 100%)]
    #text(size: 9pt)[
      *亮点*：bd 0.217→*0.614*（三轮迭代 +39.7pp）；5 类满分
      *短板*：wy 0.38（与 heidian/zmty 视觉互混）、lj 0.42（12 GT 样本最少）

      #v(0.2em)
      *根因*：弱纹理细粒度判别 + 标注碎片化——数据侧改良方向明确
    ]
  ]
]

// ============================================================
// 11 方法三：GT vs 预测可视化
// ============================================================
#slide[
  == ③ DEIMv2-L：检测效果实例（左 GT / 右 预测）

  #grid(columns: (1fr, 1fr, 1fr), gutter: 0.5em)[
    #image("assets/v103_jt.jpg", width: 100%)
    #image("assets/v103_pd.jpg", width: 100%)
    #image("assets/v103_wy.jpg", width: 100%)
  ]
  #align(center)[#text(size: 9pt)[jt 接头完美检出 / pd 纸边阴影误检 / wy 小点类别混淆——失败模式与量化指标一致]]

  #grid(columns: (1fr, 1fr, 1fr), gutter: 0.5em)[
    #image("assets/v103_hei.jpg", width: 100%)
    #image("assets/v103_zmty.jpg", width: 100%)
    #image("assets/deimv2_coco_AP_vs_GFLOPs.png", width: 100%)
  ]
  #align(center)[#text(size: 9pt)[heidian+HD 横档对齐 / zmty 碎片合并现象 / DEIMv2 官方 AP-成本曲线]]
]

// ============================================================
// 12 方法四：DEIMv2-S 动机
// ============================================================
#slide[
  == ④ DEIMv2-S：为速度过线而生

  #grid(columns: (1fr, 1fr), gutter: 1em)[
    #text(size: 9pt)[
      *动机*：Item4 要求 ≥30 FPS\@3080，L 档只有 19.5——*精度-速度必须二选一？*
      S 档 = 官方蒸馏 ViT-Tiny（192 维）+ 9.7M 参数：
      - COCO AP 50.9（L 为 56.0）
      - 计算量 ~102 GFLOPs\@1280（L 的 27%）
    ]
    #align(center)[#image("assets/deimv2_coco_AP_vs_GFLOPs.png", width: 100%)]
  ]

  #text(size: 9pt)[*实验设计*：与 103 *完全同配方*（lr 分层 / 17 类头 / Dense O2O / batch 累积），单变量 = backbone——干净的小模型对照]
]

// ============================================================
// 13 方法四：104 结果
// ============================================================
#slide[
  == ④ DEIMv2-S：速度达标 + 精度守住

  #grid(columns: (1fr, 1fr), gutter: 1em)[
    #table(
      columns: (auto, auto, auto, auto),
      align: (left, right, right, right),
      inset: 4pt,
      [*指标*], [*104-S*], [*103-L*], [*096*],
      [R\@0.25], [0.759], [*0.775*], [0.757],
      [P\@0.25], [*0.475*], [0.460], [0.403],
      [AP50], [0.696], [*0.708*], [0.676],
      [FPS\@3080], [*31.9 ✓*], [19.5 ✗], [~24 ✗],
    )
    #align(center)[
      #image("assets/fps_accuracy_scatter.png", width: 100%)
      #text(size: 9pt)[S = 唯一双达标点（绿）]
    ]
  ]
  #align(center)[#text(size: 9pt)[只掉 1.6pp 召回，换 1.6× 速度；R/P 双超 RT-DETR 基线]]
]

// ============================================================
// 14 数据管线：三条支撑线
// ============================================================
#slide[
  == 支撑线：数据管线与评测协议（贯穿四方法）

  #grid(columns: (1fr, 1fr), gutter: 1em)[
    #text(size: 9pt)[
      *6 条数据管线*：
      - 实例中心 patch 池（bd/hei 583）
      - 对比度池 810 / 小波池 1028
      - 合成池 2388（种子固定）
      - v2/v3 无泄漏划分（8:1:1 类分层）

      #v(0.3em)
      *3 套评测协议*：
      - 正式口径：self-val + conf 0.25 + IoU 0.5
      - PR 全扫描（best-F1 定位）
      - few-shot 协议（K=1/3/5/10 ×3 seeds）
    ]
    #rect(stroke: cRed + 1.2pt, radius: 4pt, inset: 6pt)[
      #text(size: 9pt)[*泄漏教训*（代价最高的坑）：
        v2 val 与 v1 train 重叠 80.9% → 跨集验证数字全部作废；
        重建无泄漏增强池后所有结论重测。
        *现在所有 run 都带数据清单快照，可精确复现*]
    ]
  ]
]

// ============================================================
// 15 四方法终局对比
// ============================================================
#slide[
  == 终局对比：R\@0.25 / best-F1 / 速度 三维

  #align(center)[#image("assets/fps_accuracy_scatter.png", width: 72%)]
  #v(0.2em)
  #table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    inset: 4pt,
    [*路线*], [*R\@0.25*], [*best-F1*], [*FPS\@3080*], [*交付建议*],
    [① YOLO 019], [0.621], [0.571\@0.47], [~62], [速度充裕·精度不足],
    [② RT-DETR 096], [0.757], [0.526 旧测], [~24], [精度好·速度临界],
    [③ DEIMv2-L 103], [*0.775*], [*0.704\@0.47*], [19.5], [精度上限·需 TRT 过线],
    [④ DEIMv2-S 104], [0.759], [0.684\@0.51], [*31.9*], [*首选交付*],
  )
]

// ============================================================
// 16 结论
// ============================================================
#slide[
  == 结论

  1. *四代实践*：YOLO 基线 0.590 → loss 加权 0.621 → RT-DETR 0.757 → DEIMv2-S *0.759（速度达标）* / DEIMv2-L *0.775（精度上限）*
  2. *配方 > 架构*：102→103 证明同架构下官方训练配方值 7pp
  3. *表征是王道*：DINOv3 自监督预训练把 bd 小目标从 0.217 拉到 0.614
  4. *工作量背书*：81 run / 6 管线 / 3 协议，每条结论都有对照实验
]
