# VR180 SBS 换脸工作台

对 180° ERP 的左右并排（Side-by-Side）VR 视频做换脸的工作台：用专为 VR180 训练的 YOLOFace 直接在两眼的 ERP 画面上定位人脸，选定目标后按固定取景抽取透视片，交给 FaceFusion 换脸，再把结果逆重映射回 SBS 输出。

> **当前为首版「固定双眼取景」**：取景视角由参考帧的目标框决定，整段复用同一取景。适合目标大致留在取景框内的镜头；目标大幅移动、转头或镜头运动时会漏换。逐帧跟踪在后续版本实现。

## 目录

- `workbench/` — 工作台本体（HTTP 服务 + 前端 + 透视/换脸/回填管线）
- `facefusion/` — 对 FaceFusion 的一处小补丁（进度/预览回传，可选）
- `docs/` — 实施计划与检测思路

## 依赖

- Python 3.10（推荐 conda 环境名为 `facefusion`）
- [FaceFusion](https://github.com/facefusion/facefusion)：提供换脸 / 增强模型与 `headless-run` 入口
- FFmpeg / FFprobe 在 PATH
- 模型权重：
  - FaceFusion 模型（inswapper 等）按 FaceFusion 文档放置
  - 粗检测模型 `yoloface_12n-vr180.onnx`，来自 [Glat0s/yolo-face 的 vr 发布](https://github.com/Glat0s/yolo-face/releases/tag/vr)，SHA-256 `8d5e9187fe5aa8f95be7a3267802639299fb39df969407d22bc379eed9aaf1e1`，放到仓库根目录的 `.assets/models/`

## 安装

1. 在同一环境安装 FaceFusion（或复用已有的 FaceFusion 目录）。
2. 把 `facefusion/workbench_progress.py` 复制到 FaceFusion 的 `facefusion/` 目录，并按 `facefusion/workbench-progress.patch` 给 4 个文件打补丁。这一步只影响进度与预览回传，**跳过也能正常换脸**，只是没有实时进度。
3. 下载 `yoloface_12n-vr180.onnx` 放到 `.assets/models/`。

## 使用

```powershell
python -B -u workbench/server.py --port 8080
```

浏览器打开 http://127.0.0.1:8080/ ：

1. 填本地路径或用文件选择器导入 VR180 ERP SBS 视频和来源脸。
2. 在片段内选一帧，检测后点选左右眼中的同一人物；未检出可手动画框。
3. 查看左右眼透视裁片，按需调整取景余量。默认处理整段视频。
4. 开始换脸，下载输出视频与带框 PNG。

分步测试入口 http://127.0.0.1:8080/test 可逐步执行「读取 → 透视导出 → 换脸 → 回填」。

## 说明

- 只针对每眼 180° ERP 的横向 SBS；浏览器预览是平面 SBS，头显自动识别依赖播放器与空间元数据。
- `server.py` 检测到当前解释器缺少 onnxruntime 时，会自动切到 `facefusion` conda 环境重启，不必手动激活。
- 生成文件写在 `workbench/.workbench/jobs`，该目录及 `.assets/` 已在 `.gitignore` 中排除。
