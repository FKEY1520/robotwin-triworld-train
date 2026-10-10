# 本次构建与验证记录

实际读取本机 `robotwin-to-triworldbench/outputs/adjust_bottle_clean50`，构建到 `datasets/train48`。以下结果来自真实数据运行，不是预估的样本数。

| 项目 | 结果 |
| --- | --- |
| 独立完整轨迹 | 50 |
| 原始三相机同步时刻 | 7,238 |
| 存储的无损 PNG | 21,714 |
| 原始图像大小 | 高 240、宽 320，RGB |
| 动作 | `joint_action/vector`，14 维，原始 float64 |
| 历史 / 未来 / 步长 | 9 / 48 / 8 |
| 全部训练窗口 | 664 |
| 验证 / 测试窗口 | 0 / 0 |
| HDF5 归档总大小 | 1,156,449,008 字节，约 1.16 GB |
| 末尾 48 帧 | 每个 episode 的最后窗口均完整覆盖 |

深度校验检查全部 HDF5 文件 SHA-256、窗口索引 SHA-256、统计文件 SHA-256、完整动作数组、窗口索引、训练角色以及全部 21,714 张图像解码。所有真实未来帧（每条轨迹的帧 1 至末帧）都至少进入一个监督窗口。动作统计按完整轨迹计算，不重复计算重叠窗口或历史补帧。

本地结果在 `datasets/train48/report.json` 和 `reports/verification_train48.json`；它们是本机生成文件，已从 Git 提交中排除。每个任务目录也有独立报告和统计。

## 可重复验证

```powershell
Set-Location -LiteralPath 'A:\Triworldbench训练数据集'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe .\convert_dataset.py verify --dataset-root .\datasets\train48 --deep
```

测试使用真实的小型 HDF5/PNG/JPEG fixture，覆盖帧与动作精确对齐、全局锚点历史、重叠窗口、尾部覆盖、跨任务独立标识、迁移并删除原始目录后的读取、统计的唯一时刻计数、可选动作字段拼接、插件接口、损坏文件及元数据拒绝、NPZ 无 pickle 读取。

核对现有模型历史规则时，可设置：

```powershell
$env:TRIVIEW_MODEL_ROOT = 'A:\Triworldbench测评模型自主开发树\Wan2.2-TI2V-5B基座_Text_3view_action原生设计'
$env:TRIWORLD_DATASET_ROOT = 'A:\Triworldbench训练数据集\datasets\train48'
.\.venv\Scripts\python.exe -m pytest -q
```

该检查只读取原模型中三个纯索引函数，不导入权重。所有 664 个真实窗口及 4,000 种历史边界组合与原模型规则一致。单文件部署提取测试也验证摘要、路径、同版本复用以及已有用户文件的保护。本机部署包中的 `train.py`、`data.py`、`history.py`、`normalization.py`、`flow.py`、`cache.py` 已与原模型项目逐文件 SHA-256 核对一致。

## 已验证范围

转换、完整真实数据校验、样本 NPZ 导出和 Python 3.11 测试已运行。GitHub Actions 配置覆盖 Python 3.10 / 3.12，远端 CI 尚需推送后执行。

本机未运行原生 T5/VAE GPU 编码或真实训练更新。提供的数据集包含真实训练图像和动作；供原模型训练器使用的 `.pt` 潜变量缓存需要在 LRZ 的原模型环境中通过 `scripts/prepare_lrz_cache.py` 生成。48 帧版本的显存占用与训练质量应以 LRZ 首次实际运行为准。详见 [LRZ 操作步骤](LRZ.md)。
