**中文** · [English](README_EN.md)

<div align="center">

# 🎭 VR180 SBS 换脸工作台

**在 180° 立体 VR 视频里，把一张脸换成另一张。**

对 180° ERP 的左右并排（Side-by-Side）VR 视频做换脸的工作台：YOLOFace 在两眼原画中定位人脸，FaceFusion 在动态透视裁片中换脸，再把结果回填到 SBS 画面。

[![GitHub stars](https://img.shields.io/github/stars/Supergatti/vr180-sbs-face-swap?style=for-the-badge&logo=github&logoColor=white&labelColor=1c2128&color=2ea44f)](https://github.com/Supergatti/vr180-sbs-face-swap/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Supergatti/vr180-sbs-face-swap?style=for-the-badge&logo=github&logoColor=white&labelColor=1c2128&color=0969da)](https://github.com/Supergatti/vr180-sbs-face-swap/forks)
[![Python](https://img.shields.io/badge/Python-3.10-3776ab?style=for-the-badge&logo=python&logoColor=white&labelColor=1c2128)](https://www.python.org/)
[![Windows](https://img.shields.io/badge/平台-Windows-0078d6?style=for-the-badge&logo=windows&logoColor=white&labelColor=1c2128)](#)

</div>

> **正式任务采用动态双眼取景。** 每帧在内存中完成裁切、换脸和回填，不生成中间透视视频。固定取景流程保留在 `/test`。

---

## ✨ 功能

| 功能 | 说明 |
| --- | --- |
| 🎯 **VR180 原生检测** | 专为本项目训练的 `yoloface_12n-vr180.onnx` 直接在 ERP 球面画面上定位人脸，无需预先展开成平面图 |
| 👁️ **双眼独立处理** | 左、右眼分别定位、取景、换脸与逆回填，同一人物使用同一来源身份 |
| 🖼️ **透视导出 / 球面回填** | 按目标框抽取透视裁片 → FaceFusion 换脸 → 逆重映射回 SBS，保留双目布局、帧时序与音频 |
| 🔄 **换脸 + 人脸增强** | 可选的人脸增强（勾选后流程由 3 步变为 4 步），可关闭以换取速度 |
| 📊 **实时进度** | 帧进度、速度、SBS 预览和保留原画帧数；可下载参考帧带框 PNG |
| **身份核查** | 无核查、宽松、严格三档。宽松允许一眼通过后补检另一眼；严格要求双眼都通过 |
| **FaceFusion 参数** | `/facefusion` 提供模型、像素增强、遮罩、识别距离和检测角度，以及参考帧试跑 |
| 🧑‍🤝‍🧑 **指定人物** | 点选左右眼中的同一人物进行替换，未检出时可手动画框兜底 |
| 🧪 **分步测试入口** | `/test` 页可逐步执行「读取 → 透视导出 → 换脸 → 回填」，便于排查 |
| 🎚️ **推理设备切换** | CUDA · GPU / CPU 一键切换 |

## 目录

- `workbench/` — 工作台本体（HTTP 服务 + 前端 + 透视/换脸/回填管线）
- `facefusion/` — 动态人脸分析兼容补丁与分步测试进度回传补丁
- `scripts/` — 依赖安装与权重下载脚本
- `docs/` — 实施计划与检测思路

## 一键安装（Windows）

前置：已安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)（或 Anaconda）、[Git](https://git-scm.com/)，且 `ffmpeg` / `ffprobe` 在 PATH。

在仓库根目录打开 PowerShell，运行：

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

脚本会依次：

1. 创建（或复用）conda 环境 `facefusion`（Python 3.10）；
2. 把 FaceFusion **3.1.2** 浅克隆到 `.facefusion/` 并安装其依赖（含 onnxruntime-gpu）；
3. 自动下载本工作台专有的 VR180 检测模型 `yoloface_12n-vr180.onnx`（SHA-256 校验）；
4. 应用人脸分析兼容补丁，尝试应用分步进度补丁，并准备默认检测、换脸、增强和遮罩模型。

FaceFusion 权重放在 `.facefusion/.assets/models/`；额外模型安装后，刷新 `/facefusion` 即可选择。

## 启动

```powershell
powershell -ExecutionPolicy Bypass -File start.ps1
```

（Windows 资源管理器里也可双击 `start.bat`。）

浏览器打开 http://127.0.0.1:8080/ ：

1. 填本地路径或用文件选择器导入 VR180 ERP SBS 视频和来源脸。
2. 在片段内选一帧，检测后点选左右眼中的同一人物；未检出可手动画框。
3. 查看左右眼透视裁片，按需调整取景余量。默认处理整段视频。
4. 开始换脸，下载输出视频与带框 PNG。

分步测试入口 http://127.0.0.1:8080/test 可逐步执行「读取 → 透视导出 → 换脸 → 回填」。

高级参数入口 http://127.0.0.1:8080/facefusion 可先试跑参考帧。检测默认只用 0°；加选 90°、180°、270° 会应用到粗检和精检，也会增加推理耗时。

### 推理设备

「推理设备」下拉可选：

- **CUDA · GPU** — onnxruntime 的 CUDAExecutionProvider，最稳，默认。
- **CPU** — 无 GPU 时兜底。

## 手动安装（不推荐）

1. 在同一环境安装 FaceFusion 3.1.2（或复用已有的 FaceFusion 目录，将其根目录路径写入环境变量 `FACEFUSION_HOME`）。
2. 在 FaceFusion 根目录应用本仓库的 `facefusion/workbench-analyser.patch`（动态换脸必需）。分步进度可选：复制 `facefusion/workbench_progress.py` 并应用 `facefusion/workbench-progress.patch`。
3. 在 FaceFusion 的 `.assets/models/` 安装所选模型；将 VR180 检测模型放到工作台根目录的 `.assets/models/`，或运行 `python scripts/download-models.py` 获取它。

## 说明

- 只针对每眼 180° ERP 的横向 SBS；浏览器预览是平面 SBS，头显自动识别依赖播放器与空间元数据。
- `server.py` 检测到当前解释器缺少 onnxruntime 时，会自动切到 `facefusion` conda 环境重启。
- 透视抽取/逆回填的重映射优先使用 `torch.cuda`，其次是 `cv2.cuda`，都没有则退回 CPU（仍可换脸，仅更慢）。可选安装带 CUDA 的 torch 以启用 GPU 重映射。
- 生成文件写在 `workbench/.workbench/jobs`；`.assets/`、`.workbench/`、`.facefusion/` 已在 `.gitignore` 中排除。

---

## ⭐ 星标

喜欢这个项目？点个星标支持一下：

[![Star this repo](https://img.shields.io/badge/Star-给个星标-2ea44f?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Supergatti/vr180-sbs-face-swap)

**星标历史**（在仓库转为公开后会开始累计）：

[![Star History Chart](https://api.star-history.com/svg?repos=Supergatti/vr180-sbs-face-swap&type=Date)](https://star-history.com/#Supergatti/vr180-sbs-face-swap&Date)
