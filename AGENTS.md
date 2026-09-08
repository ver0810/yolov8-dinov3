# Repository Guidelines

## Project Overview

织物缺陷检测实验项目（17 类：lj 垃圾 / bd 白点 / heidian 黑点 / wy 污印 / zyc 纸异常 / zmty 脏面条印 / cq 串气 / zj 纸接 / bj 布接 / jt 接头 / yy 压印 / bmss 表面损伤 / zy 皱印 / pd 破洞 / HD 横档 / cy 重影 / jiaodai 胶带）。约束：**只交训练权重、禁用推理端手段**（TTA/高分辨率推理/集成）。目标：训练侧正收益探索 + 竞争评分（Item1 召回≥95% / Item2 过检≤5% / Item3 few-shot / Item4 ≥30fps / Item5 报告）。

**正式评测口径（已落板）**：
- **self-val**：每个 run 用自己 `args.yaml` 的 `data` 划分做 val（v1 模型 → v1 val；v2 模型 → v2 val）；**禁止跨集验证**（v2 val 与 v1 train 重叠 80.9% = 泄漏，已坐实）
- **conf=0.25 主基准**（+ IoU 0.5），best-F1 参考；`maxR/peakR`（conf→0 兜底）与旧 0.8x 高分**全部作废**
- 推理分辨率：**1280**（用户 2026-08 落板；训练 1280 的模型只能在 1280 推理，640/1536+/2048 均失配崩溃；640 训练模型只能在 640 推理）

## Architecture & Data Flow

```
训练: configs/own_dataset{,_v2}.yaml → scripts/train.py / /tmp/train_*.py（hub start 后台）
     → outputs/runs/0XX_*/weights/{best,last}.pt
数据池: 整图(v1 train 1445) + 实例中心 patch(bd/heidian 583) + 合成池(2388) + wavelet 池(2985)
     → outputs/dataset_v1_patch_*/train_pool*.txt
评测: scripts/eval_scoring.py（正式口径） / /tmp/perclass_argv.py（逐类 R@0.25 @1280）
     → outputs/report_*.csv / final_selfval_delivery.csv
报告: presentations/03_experiment_validation/main.typ → typst compile → slides.pdf
```

**loss 加权补丁**（`third_party/ultralytics/ultralytics/utils/loss.py`，v8DetectionLoss 内，环境变量门控、默认关闭）：
- `SMALL_DEFECT_CLS="1,2"` + `SMALL_DEFECT_GAIN`：类级 cls 权重（bd/heidian）
- `SMALL_DEFECT_AREA`：面积感知加权（小目标 cls loss ×gain）
- `SMALL_DEFECT_NEG` + `SMALL_DEFECT_NEG_GAMMA`：**难例感知背景加权** `w = 1 + (NEG-1)·p^γ`（注意：`NEG·p^γ` 公式是错的——p<1 会低于 1 反向放松背景，曾导致 066 崩溃 54300 框）

## Key Directories

| Path | Purpose |
|---|---|
| `scripts/` | 训练/评估/数据管线（train.py、eval_scoring.py、eval_fewshot*.py、make_patch_bdhd.py、make_synth_{bdhd,wavelet}.py、make_bg_patch.py、stratified_split.py…） |
| `configs/` | 数据划分 yaml（own_dataset.yaml=v1、own_dataset_v2.yaml=v2）+ exp_*.yaml 实验配置 |
| `outputs/runs/` | 实验 001–068+（weights/、args.yaml、results.csv）；028 缺失；006≡055、046≡047 为重复 run |
| `outputs/dataset_v2/` | v2 划分数据（train 1277 / val 350 图 / 678 GT） |
| `outputs/dataset_v1_patch_bdhd{,_synth,_wavelet,bg}/` | patch/合成/背景池（供 061+ 训练） |
| `outputs/` 根 | 评测产出：report_retest_selfval_conf025.csv、final_selfval_delivery.csv、perclass_057_conf025.txt、final_submission_weights.pt |
| `presentations/03_experiment_validation/` | 报告源（main.typ）+ slides.pdf + per_class_map.md（mAP/速度记录） |
| `third_party/` | vendored ultralytics（8.4.104，含 loss.py 补丁）、dinov3 |

## Development Commands

```bash
uv run python scripts/eval_scoring.py --weights <best.pt> --data configs/own_dataset_v2.yaml --imgsz 1280  # 正式口径评分
uv run python /tmp/perclass_argv.py outputs/runs/0XX_*/weights/best.pt   # 逐类 R@0.25 @1280（v1 val）
uv run python /tmp/perclass_map_1280.py <weights>                        # 逐类 mAP50 @1280
typst compile presentations/03_experiment_validation/main.typ            # 编译报告
uv run --with pywavelets python scripts/make_synth_wavelet.py            # wavelet 合成池（依赖 pywavelets）
```

**训练启动**（长任务必须用 hub start，勿用 nohup——nohup 进程会被静默清理）：
```
hub start name=XXX application=uv args=[run, python, /tmp/train_XXX.py] cwd=项目根
```
训练脚本惯例：`YOLO(<pretrained>)` + `.train(data=..., imgsz=1280, batch=4, epochs=60, lr0=0.01, optimizer="auto", project="<绝对路径>/outputs/runs", name="0XX_...")`。**project 必须绝对路径**（相对路径会落到 `runs/detect/...`）。

## Code Conventions & Common Patterns

- **实验编号**：runs 目录 `0XX_模型_变体`（058+ = patch 系列）；评估对照锚点：004（v1 基线 0.6140）、019（1280+cls_w 0.6213）、063（合成池 bd 0.4773/heidian 0.3333 @1280）、068（组合）
- **评测脚本**：`/tmp/perclass_argv.py` 接受 argv 权重路径（勿用 sed 改硬编码路径——历史上多次 sed 静默失败导致假数据）；GT 计数从 `/home/ancheng/dataset/dataset_split/val/labels`（v1）或 `outputs/dataset_v2/val/labels`（v2）统计（`box.nt_per_class` 在该版本不存在）
- **口径纪律**：R@0.25 = conf≥0.25 且 IoU≥0.5；宏平均 = 17 类逐类平均；对比必须同推理分辨率
- **loss 补丁**：只改 `third_party/ultralytics/utils/loss.py` 的 v8DetectionLoss cls 段，环境变量门控，可复现记录
- **合成/切图脚本**：固定 random seed（7/11/21），输出绝对路径 pool txt + 空/完整标签；切图丢弃 >patch 尺寸目标（切破即丢，不保留部分）
- **PPT（Typst）**：`<95%` 需转义 `\<95%`；`R@0.25` 用 `R\@0.25`；页码引用按 18 页布局

## Important Files

| File | Why it matters |
|---|---|
| `configs/own_dataset.yaml` / `own_dataset_v2.yaml` | v1/v2 划分（训练/评测数据源） |
| `third_party/ultralytics/ultralytics/utils/loss.py` | 小缺陷 loss 加权补丁（环境变量门控） |
| `scripts/eval_scoring.py` | 正式评分（conf=0.25 主基准 + best-F1；旧 maxR 已降级为诊断） |
| `scripts/eval_fewshot.py` + `eval_fewshot_scoring.py` | few-shot 协议（`--train-dir outputs/dataset_v2/train` 必须显式覆盖默认 v1 路径） |
| `scripts/make_patch_bdhd.py` | 实例中心 patch 池（bd/heidian 专用，尺寸分流策略） |
| `outputs/report_retest_selfval_conf025.csv` | 57 模型 self-val 原始数据 |
| `outputs/final_selfval_delivery.csv` | 55 唯一模型排序交付表（rank/R@0.25/P@0.25/F1/peakR 参考） |
| `presentations/per_class_map.md` | 逐类 mAP + 1280 口径 + 速度记录 |
| `presentations/03_experiment_validation/main.typ` | 18 页实验验证报告源 |

## Runtime/Tooling Preferences

- 运行时：**uv**（`uv run python ...`）；Python 3.11；GPU RTX 3060 Laptop **6GB**（1280 训练 batch≤4、640 batch16；P2 结构 1280 仅 batch4 且 17min/ep）
- 后台任务：**hub start**（受保护，可 wait/logs/stop）；不要 nohup（进程会无声死亡）
- 原图 2048×2048（v1 train 1445 图 / val 181 图 377 GT；v2 train 1277 / val 350 图 678 GT）
- 阈值/损失相关可选环境变量：SMALL_DEFECT_CLS/GAIN/AREA/NEG/NEG_GAMMA（见 loss.py 补丁）
- 评测 GPU 争用：训练运行中并发 val 会慢 10×（避免并行评测）
## Testing & QA

- **无正式测试框架**；验证 = 运行实验 + 官方口径评估（self-val @1280，conf=0.25）
- 关键对照（当前状态）：004=0.6140 / 019=0.6213（v1 组）；045=0.5476 / 044=0.5316（v2 组）；063=0.6367 宏 / bd 0.4773 / heidian 0.3333（当前最强召回）；065–067 背景抑制实验全被证伪（NEG 伤 bd、背景 patch 伤 heidian）
- **070 难例背景（1280，063 last 迁移+819 难例背景 patch+20ep）：失败**——同口径对比 063 R@0.25 0.719→0.606、FP 率 50.2%→46.3%；FP 降 4pp 但召回降 11pp（bd 0.409→0.273、zmty 0.432→0.270）。结论：**FP 高≠纯背景，val 标注稀疏藏真缺陷，压背景=压真阳性**；背景抑制所有 4 形态（065/066/067/070）均负收益，方向关闭
- **071 背景注入受控对比（640，唯一变量=27% 无缺陷背景，38ep 同配置）**：071c（组合池）宏 R 0.546/FP 40.3% vs 071d（+2000 背景）宏 R 0.498/FP 38.1%——FP 降 2.2pp 但 R 降 5pp（zy 0.519→0.333）→ **第四种背景形态确认负收益**；且组合池@640 bd 0.068 < 004 0.114（patch 增益与 1280 绑定，640 下消失）
- **类间混淆诊断（063@1280）**：错配对集中在弱纹理组 {heidian→wy 6、bmss→wy/zmty 6、lj→bmss/zyc 4、zmty→bmss/zy 4}——视觉相似类样本少（train wy 363/bmss 378/zmty 267/zy 217/lj 126 实例），无合成增强
- **增强参数验证（063 P3 特征，2026-08-13）**：亮度±15%/对比度0.8–1.3/旋转±15°/缩放0.8–1.2 的变体**全部保留类簇**（intra≥0.795 vs inter≤0.516，全部 OK）——增强不破坏类别可分性，可安全扩展到弱纹理组
- **已知陷阱**：
  - 跨集验证 = 泄漏（80.9% 重叠），任何跨划分数字作废
  - 1280 训练模型 ≠640 推理（004@1280 宏 0.309 崩）；≠1536/1792/2048（超 scale 增强上限 1920 失配）
  - sed 改评测脚本路径易静默失败 → 用 argv 传参 + 事后核对输出模型名
  - 完成态 ckpt 的 epoch=-1，resume 需手动 patch（`ckpt["epoch"]=N` + `train_args["epochs"]=M`）后 `train(resume=path)`
  - 训练内 1280-val 选出的 best.pt 在 640/1280 正式口径下可能非最优 → 双测 best/last
- 数据记录规范：结果表写 `presentations/per_class_map.md`（YAML front matter 元数据：date/split/metric/models/sources）
