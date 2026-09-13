# Classroom Recognition Service

独立的「课堂答题与学情自动化系统」子项目：只负责把一张答题页面图像转换成可解释的结构化识别结果。

当前范围：

- 固定模板页面检测、四角定位点检测、透视校正和标准尺寸还原；
- 基于 OpenCV + NumPy 的矩形信息点 OMR；
- 可替换 OCR Provider 接口，首选 PP-OCRv5；
- 页面质量、识别置信度、空白/双选/浅涂/歧义和复核状态；
- FastAPI `POST /recognize/page` 与 OpenAPI；
- 同一 Crop 的 OCR Benchmark Harness。

不包含 Web 管理后台、课堂 Dashboard、机构/班级/学生业务、Android、APK、SSE、家长反馈、正式判分或标准答案。Recognition Service 不接收答案键，也不返回对错和分数。

## Directory

```text
src/                         正式服务代码
templates/                   版本化固定答题卡模板
_research/upstream/           OMRChecker / DeepSeek-OCR-2 对照快照
benchmarks/                  OCR 基准工具
datasets/synthetic/          非 PII 合成样本生成器与样本目录
datasets/local-private/      本地私有样本，Git 忽略
tests/                       自动化测试
docs/                        API、上游研究、Provider、硬件和 Benchmark 说明
```

## Local setup

PowerShell：

```powershell
$project = 'D:\Codex\01_App-应用开发\005_ClassroomRecognitionService-课堂答题识别服务'
Set-Location $project
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts\generate_synthetic_samples.py
.\.venv\Scripts\python.exe -m pytest
```

最小安装不强制下载 OCR 模型；没有 PaddleOCR 时，OMR 仍可运行，文本题会返回 `REVIEW_REQUIRED`。启用 PP-OCRv5：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[paddle]"
$env:PADDLEOCR_MODEL_NAME = 'PP-OCRv5_server_rec'
$env:PADDLEOCR_DEVICE = 'cpu'
```

启动服务：

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8092
```

## HTTP example

```powershell
curl.exe -X POST http://127.0.0.1:8092/recognize/page `
  -F "image=@datasets/synthetic/answer-sheet-a-v0.1-recognized.png;type=image/png" `
  -F "template_version=A-v0.1" `
  -F "page_type=OMR"
```

详见 [API Contract](docs/API_CONTRACT.md)。

## Recognition policy

- 模板坐标只存在于 JSON，不散落在 Python 源码中；
- OMR 输出 `fill_ratios`、top/second score、gap、阈值配置版本和状态；
- 不确定时不猜答案；
- OCR 的 `raw_text` 保留引擎原文，`normalized_text` 只做首尾空白清理，不做拼写纠正；
- DeepSeek Vision 只有同时配置 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_VISION_MODEL` 才启用，且只发送单个文字 ROI Crop；
- TrOCR 与 DeepSeek-OCR2 仅为 Benchmark Candidate，不是生产默认 Provider。

## Current validation state

已实现并可在本机自动验证：合成透视页面的 `1 B / 2 C / 3 A / 4 D`、空白、双选、浅涂、歧义、页面质量失败、HTTP multipart 上传和 OpenAPI。PP-OCRv5、TrOCR、DeepSeek-OCR2 的真实模型运行取决于可选依赖/模型/凭据，缺失时明确记录为未运行或复核，不伪报通过。
