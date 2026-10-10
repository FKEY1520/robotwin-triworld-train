# RoboTwin → TriWorldBench 自研模型 1.0 训练数据构建

将 `robotwin-to-triworldbench/outputs` 中的 RoboTwin 任务转换为可以搬到 LRZ 的训练数据。每个任务独立生成一份数据集，同时提供汇总训练入口。所有样本均属于训练集；本项目不创建验证集。

默认目标是预测未来 **48 帧**。输入使用与模型匹配的 **9 帧历史**：初始锚点帧 + 最近 8 帧。窗口步长为 **8 帧**，并额外保留最后一个完整窗口，使视频最后 48 帧进入训练。原始 RGB 图片和动作按 episode 保存一次，重叠窗口仅增加索引，避免反复复制视频。

当前 `adjust_bottle_clean50` 已生成到 `datasets/train48`：

| 项目 | 当前结果 |
| --- | --- |
| episode 数 | 50 |
| 原始时间帧数 | 7,238 |
| 三视角 PNG 图片数 | 21,714 |
| 未来 48 帧训练窗口数 | 664 |
| 验证 / 测试样本数 | 0 / 0 |
| 原始动作精度 | float64，归档保留原值 |
| episode 归档总大小 | 1,156,449,008 字节，约 1.16 GB |

结果可在 `datasets/train48/report.json` 查看。下方重建命令使用 `train48_v2`，已有 `train48` 不会被覆盖。

## 1. 项目目录

```text
Triworldbench训练数据集/
├── convert_dataset.py          # 命令行入口
├── triworld_train/             # 转换、窗口、归一化、读取和校验实现
├── configs/conversion48.json   # 默认 48 帧转换参数
├── configs/lrz_train48.toml    # LRZ 训练配置参考
├── examples/custom_transform.py# 修改提示词、动作等的扩展接口
├── scripts/prepare_lrz_cache.py# 生成现有 LRZ 训练代码可读取的缓存
├── tests/                     # 有实际数据语义的自动测试
├── docs/                      # 格式、定制、LRZ 和 GitHub 使用说明
└── datasets/train48/          # 生成的数据；不提交 GitHub
```

详细文档：[数据格式](docs/DATA_FORMAT.md)、[修改数据与参数](docs/CUSTOMIZATION.md)、[LRZ 接入](docs/LRZ.md)、[验证记录](docs/VERIFICATION.md)、[上传 GitHub](docs/GITHUB.md)。

## 2. 安装运行环境

要求 Python 3.10 或更新版本。数据转换只需要 NumPy、h5py 和 Pillow；不需要 GPU 或 PyTorch。模型训练使用 LRZ 已有的 PyTorch 环境。

Windows PowerShell：

```powershell
Set-Location -LiteralPath 'A:\Triworldbench训练数据集'

# 已有可用虚拟环境时保留它。
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
  py -3 -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\convert_dataset.py --help
```

以下命令直接调用虚拟环境 Python，无需激活环境，也不会遇到 PowerShell 的脚本执行策略问题。

Linux / LRZ：

```bash
cd /path/to/robotwin-triworld-train
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python convert_dataset.py --help
```

也可以安装为命令行工具：`python -m pip install -e .`，随后使用 `triworld-train`；它和 `python convert_dataset.py` 功能相同。

## 3. 查看当前源数据

当前源目录：

```text
A:\robotwin_to_Triworldbench_github_ALOHA\robotwin-to-triworldbench\outputs
```

先检查一个 episode 的动作、图片数量和任务信息：

```powershell
.\.venv\Scripts\python.exe .\convert_dataset.py inspect `
  --source-root 'A:\robotwin_to_Triworldbench_github_ALOHA\robotwin-to-triworldbench\outputs' `
  --tasks adjust_bottle_clean50 `
  --limit 1
```

`--tasks` 选择源目录中的任务包名称。省略它会扫描全部可识别的任务包。`--limit` 限制每个任务本次读取的 episode 数量，适合先做小规模检查；正式构建时省略该参数，或设为 0。

`inspect` 检查字段、帧数和图片文件索引，不解码所有图片。正式转换会逐张解码检查 RGB 及分辨率，构建后还可使用 `verify --deep` 检查归档。

## 4. 构建当前任务的完整训练集

```powershell
.\.venv\Scripts\python.exe .\convert_dataset.py convert `
  --source-root 'A:\robotwin_to_Triworldbench_github_ALOHA\robotwin-to-triworldbench\outputs' `
  --output-root '.\datasets\train48_v2' `
  --tasks adjust_bottle_clean50 `
  --config '.\configs\conversion48.json'
```

或者明确给出参数（以上两种构建方式选择一种）：

```powershell
.\.venv\Scripts\python.exe .\convert_dataset.py convert `
  --source-root 'A:\robotwin_to_Triworldbench_github_ALOHA\robotwin-to-triworldbench\outputs' `
  --output-root '.\datasets\train48_v2' `
  --tasks adjust_bottle_clean50 `
  --horizon 48 --stride 8 --history-frames 9 `
  --image-source auto --action-key joint_action/vector --action-dim 14
```

默认 `auto` 优先选择每个 episode 完整的无损 PNG 序列；PNG 不完整时使用完整 JPEG 序列。一个 episode 内不会混合 PNG 和 JPEG。`report.json` 会记录实际选择及处理情况。

转换器拒绝覆盖已经存在的输出目录。重复构建时将 `--output-root` 改为新的版本目录，例如 `datasets/train48_v2`。这样可以保留原数据及其训练统计。

生成结构（下方为已构建的 `train48`；新版本目录结构相同）：

```text
datasets/train48/
├── dataset.json
├── windows.jsonl
├── normalizer.json
├── report.json
└── adjust_bottle_clean50/
    ├── dataset.json
    ├── windows.jsonl
    ├── normalizer.json
    ├── report.json
    └── episodes/*.h5
```

汇总目录用于联合训练多个任务；单个任务目录用于独立训练或单独搬运。生成数据包含图片本身，移动后读取不依赖原始 Windows 路径。

## 5. 校验与取一个样本

```powershell
.\.venv\Scripts\python.exe .\convert_dataset.py verify `
  --dataset-root '.\datasets\train48' --deep

# 可选：导出一个窗口，便于检查数组形状和数值。
.\.venv\Scripts\python.exe .\convert_dataset.py export-npz `
  --dataset-root '.\datasets\train48' `
  --output-root '.\exports\sample48' --limit 1
```

`verify` 检查数据结构和窗口语义；`--deep` 进一步解码图片。命令失败会返回非零退出码（2），应先查看报错和生成报告。NPZ 会展开窗口中的图像，适合检查或对接其他训练代码；训练时直接读取归档可以减少重复存储。

Python 读取：

```python
from triworld_train.dataset import WindowDataset

dataset = WindowDataset("datasets/train48", image_size=(256, 320))
sample = dataset[0]
print(sample["history_video"].shape)  # (3, 3, 9, 256, 320)
print(sample["future_video"].shape)   # (3, 3, 48, 256, 320)
print(sample["packed_actions"].shape) # (57, 14)
print(sample["prompt"])
```

三视角顺序为 `head, left, right`，颜色为 RGB。图像输出为 `float32`，范围 `[-1, 1]`。图像大小参数顺序是 `(高度, 宽度)`。PyTorch 环境可使用 `as_torch=True`。更多模型字段和现有 LRZ 训练缓存的生成方法见 [LRZ 接入](docs/LRZ.md)。

## 6. 下载新任务后的操作

把新任务包放入同一个 `outputs` 目录，并保留它的原始结构，然后运行 `inspect` 检查。

为新任务生成数据时，指定新任务的目录名：

```powershell
# 把 new_task_clean50 替换成实际任务包目录名。
.\.venv\Scripts\python.exe .\convert_dataset.py convert `
  --source-root 'A:\robotwin_to_Triworldbench_github_ALOHA\robotwin-to-triworldbench\outputs' `
  --output-root '.\datasets\new_task48' `
  --tasks new_task_clean50 `
  --config '.\configs\conversion48.json'
```

上述命令会在 `datasets/new_task48` 生成该任务的数据及汇总入口。转换器不向现有目录追加数据。如需联合训练新旧任务，省略 `--tasks`，使用新的输出目录，例如 `datasets/train48_all_v2`，从全部源任务重新构建汇总版本。改变预测长度、动作定义或转换插件时，也应使用新版本目录，使旧训练记录保持可追溯。

## 7. 修改数据

参数文件可以调整未来帧数、窗口步长、历史长度、图片来源和动作字段。修改提示词或动作数值时，可使用 Python 插件：

```powershell
.\.venv\Scripts\python.exe .\convert_dataset.py convert `
  --source-root 'A:\robotwin_to_Triworldbench_github_ALOHA\robotwin-to-triworldbench\outputs' `
  --output-root '.\datasets\train48_custom' `
  --tasks adjust_bottle_clean50 `
  --config '.\configs\conversion48.json' `
  --transform '.\examples\custom_transform.py:customize_episode'
```

完整接口和实例见 [修改数据与参数](docs/CUSTOMIZATION.md)。48 帧指源视频连续帧，不假设源数据的帧率；若没有可确认的 FPS，数据中将它保留为空。

## 8. 测试与 GitHub

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

GitHub Actions 会在 Python 3.10 和 3.12 上运行测试。代码、配置、示例和说明可上传 GitHub；生成数据、模型权重、环境和缓存已加入 `.gitignore`。上传步骤和数据交付方法见 [上传 GitHub](docs/GITHUB.md)。
