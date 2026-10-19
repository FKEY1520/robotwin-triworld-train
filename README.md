# TriWorldBench 风格数据的训练窗口构建工具

这个项目读取**已经转换成 TriWorldBench 风格目录的 RoboTwin 任务包**，进一步构建用于自研模型训练的三视角视频与动作窗口。默认每个样本包含 **9 帧历史、48 帧连续未来目标**，窗口步长为 **8 帧**。历史图像、任务指令及对齐的动作序列作为条件，其中未来动作也是已知条件；未来图像是监督目标。全部输出样本标记为训练集，现有验证集单独保留。

完整流程分为两个阶段：

```text
原始 RoboTwin 数据
    ↓ 第一步：使用 robotwin-to-triworldbench 转换格式
TriWorldBench 风格任务包：映射、动作、指令、三视角图片
    ↓ 第二步：使用本项目构建训练窗口
历史图像 + 已知动作条件 + 未来图像目标 + 窗口索引
    ↓ 由训练代码按相同的数据约定读取
模型训练
```

还没有第一步输出时，先使用 [RoboTwin → TriWorldBench 格式转换工具](https://github.com/FKEY1520/robotwin-to-triworldbench)。本仓库公开数据构建、校验和读取代码，数据格式对齐自研模型的训练约定；模型代码计划后续单独发布。

第一次使用，先完成环境安装并准备完整转换包。**转换包分散在整个项目目录时，推荐使用 `batch` 递归批量构建**；仅试构建或处理一个确定的任务包时，可使用 `convert`。两种方式都不要求 GPU。

## 目录

- [0. 推荐：扫描整个项目并批量构建](#batch-build)
- [1. 先理解几个概念与支持范围](#basics)
- [2. 安装项目与依赖](#installation)
- [3. 准备已经转换好的任务包](#source-data)
- [4. 完成第一次训练集构建](#first-build)
- [5. 理解历史、未来和末尾窗口](#window-rules)
- [6. 看懂输出与 Python 读取](#output-data)
- [7. 新增任务与多任务训练](#multiple-tasks)
- [8. 修改帧数、动作与数据内容](#customization)
- [9. 完整命令行参数](#cli-options)
- [10. 常见问题与排查](#troubleshooting)
- [11. 测试、验证记录与适用范围](#verification)
- [12. 更新代码并保留本地数据](#update-project)

<a name="batch-build"></a>

## 0. 推荐：扫描整个项目并批量构建

完成[环境安装](#installation)后，从本项目根目录执行。`--source-root` 可以指向上一步转换工具的整个项目，而不必逐个指定 `outputs` 中的任务。下面假设两个项目仍放在并列目录：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py inspect --source-root "../robotwin-to-triworldbench" --recursive --config "./configs/conversion48.json" --limit 0
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py batch --source-root "../robotwin-to-triworldbench" --output-root "./datasets/all_tasks_train48" --config "./configs/conversion48.json" --limit 0
```

第一条递归检查任务包；第二条实际构建。`inspect --recursive` 是严格预检，遇到首个错误会停止；`batch` 才会按包记录失败并继续处理其他包。`--limit 0` 是默认值，表示每个任务包的全部映射轨迹，不限制任务数。默认配置仍为 **9 帧历史、48 帧未来、步长 8**，末窗使用每条视频的最后 48 个真实帧。

例如，项目中存在 `adjust_bottle_clean50`、`beat_block_hammer_clean50`、`beat_block_hammer_randomized500` 时，批量命令会分别处理这些包；还有其他有效包时也会一起发现。名称及其中的数字不决定实际轨迹数，结果以本次扫描和 `batch_report.json` 为准。

`batch` 在找到 `robotwin_episode_mapping.json` 后，按这个包的映射、指令、数值和完整图片序列构建，随后停止深入该包。它跳过 Git、虚拟环境等基础目录，也不会把包内保留的 `source_data` 再识别为新任务。**扫描整个项目不等于把任意文件都转成训练样本**：原始 HDF5、脚本、压缩包和本项目已经生成的训练窗口目录都不是源转换包。

每个任务包单独构建和保存。一个包失败时，批处理记录错误并继续其他包，已完成任务保留在输出目录中。查看 `batch_report.json` 可了解完成/失败情况和轨迹、窗口计数；存在失败时命令返回非零退出码（2）。只有本次所选任务全部成功，根目录才发布可联合读取的 `dataset.json` 等汇总文件。

中断后或修复失败源包后，使用**相同源根目录、输出目录及数据配置**重新运行，并加 `--resume`：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py batch --source-root "../robotwin-to-triworldbench" --output-root "./datasets/all_tasks_train48" --config "./configs/conversion48.json" --limit 0 --resume
```

`--resume` 只复用本工具管理的批量输出，会核对源、配置及已完成任务的校验记录，再跳过有效完成项。源检查包括映射、指令和 HDF5 摘要，以及图片大小和修改时间；已完成归档还会重新核验摘要。已完成源包发生变化或输出损坏时，该包会报错，不会自动覆盖；配置变化会拒绝继续。修复失败包后可继续，修改已完成数据或数据定义则使用新输出版本。它也不能把旧 `convert` 输出直接当作批量检查点。

续跑以任务包为单位：已经完成的包复用，中断时尚未完成的包重新构建。正常退出或按 `Ctrl+C` 会释放运行锁；断电、强制结束进程后若提示输出被锁定，先确认原进程已经停止，再删除该输出根目录中的 `.batch.lock`，然后加 `--resume` 重试。不要在旧进程仍运行时删除锁。

全部成功后执行深度校验：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py verify --dataset-root "./datasets/all_tasks_train48" --deep
```

源目录和输出目录必须位于彼此的目录树之外；例如源为相邻的转换项目，输出为本项目的 `datasets/`。不要把输出放到被扫描的源项目里面。`batch` 不监视下载目录：以后加入新包，仍需手动用 `--resume` 再运行，相关更新流程见[第 7 节](#multiple-tasks)。

| 运行方式 | 扫描范围 | 出错与再次运行 |
| --- | --- | --- |
| `convert` | 指定包，或源根的直接子包。 | 本批原子构建；任一失败整批停止；需要不存在的新输出目录。 |
| `batch` | 默认递归扫描整个源项目。 | 按包保存进度并继续处理其他包；已有批量输出使用 `--resume` 校验后继续。 |

<a name="basics"></a>

## 1. 先理解几个概念与支持范围

| 名称 | 在本项目中的含义 |
| --- | --- |
| 任务（task） | 一类机器人操作，例如 `adjust_bottle`。 |
| 转换包（bundle） | 上一步生成的完整目录，例如 `adjust_bottle_clean50`；递归扫描中可用相对于源根的路径选择。 |
| 轨迹（episode） | 一次任务执行记录，包含时间对齐的动作和三视角图片。 |
| 时间帧数 `T` | 一条轨迹的同步时刻数；三视角各有 `T` 张图片，共 `3T` 张。 |
| 历史长度 `P` | 每个样本输入的历史帧数，对应 `--history-frames`，默认 9。 |
| 未来长度 `H` | 每个样本连续未来图像目标的帧数，对应 `--horizon`，默认 48。 |
| 窗口步长 `S` | 相邻常规窗口的未来起点间隔，对应 `--stride`，默认 8。 |
| 窗口（window） | 同一轨迹的 `P` 帧历史、`H` 帧未来图像及对应已知动作条件。 |
| HDF5 | 保存多组数组和图片字节的容器；源文件通常为 `.hdf5`，本项目归档为 `.h5`。 |

默认输入约定是三视角 `head, left, right`、14 维动作字段 `joint_action/vector`。图片需要是 RGB，同一条轨迹内所有视角、所有帧的分辨率一致；本项目归档时保留原分辨率。

源 HDF5 的**全部数值字段**都需要逐帧对齐，且数值不能含 NaN、无穷大或复数，包括未选择的动作字段和源 `joint_action/vector`。每条轨迹必须有有效指令，以及完整且连续编号的图片序列。只有首帧、MP4 或一份动作文件，都不足以构建本项目的数据集。

递归预检和批量模式还会检查 `test_dataset/data/` 的 episode HDF5 是否与映射一致；存在未映射的额外 episode 文件时会报错。先检查上一步转换是否完整，不能靠扫描任意 HDF5 自动补建任务。

当前训练数据约定要求：`H` 为正的 4 的倍数，`P` 为 `4n+1`（1、5、9、……），`1 ≤ S ≤ H`，每条轨迹至少有 `H+1` 帧。默认需要 **至少 49 帧**，历史不足可以重复帧 0，未来目标始终使用真实帧。

本项目不自动创建验证集或测试集。`split` 始终为 `train`；已有外部验证集不参与这里的动作统计。转换成功表示训练数据构建完成，模型训练和训练效果需要由后续训练流程验证。

输入中的 `test_dataset` 是上一步的目录命名，不决定这里的样本角色。请只选择本次准备用于训练的自有数据包，验证数据继续独立保存。

<a name="installation"></a>

## 2. 安装项目与依赖

需要 **Git、Python 3.10 或更新版本**。CPU 即可完成构建和校验，运行依赖为 `numpy`、`h5py`、`Pillow`，不需要安装 PyTorch 或启动 RoboTwin 仿真器。安装依赖后，转换本地数据不需要联网下载模型。

### Windows：PowerShell

在准备存放代码的上级目录执行：

```powershell
git -c core.autocrlf=false clone https://github.com/FKEY1520/robotwin-triworld-train.git robotwin-triworld-train
Set-Location ".\robotwin-triworld-train"
git config core.autocrlf false
py -3 --version
if (-not (Test-Path -LiteralPath ".venv")) { py -3 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import sys, numpy, h5py, PIL; print(sys.version); print('Dependencies OK')"
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py --help
```

先确认 Python 版本不低于 3.10。没有 `py` 时，可用 `python --version` 确认版本，再把创建环境命令中的 `py -3` 换为 `python`。

已经有 `.venv` 时，先运行 `.\.venv\Scripts\python.exe --version` 检查；**保留现有环境，不要用另一个 Python 版本覆盖创建**。确实需要换版本时，先将旧环境移到仓库外备份，再创建新的空白 `.venv` 并安装依赖。不要复制旧环境的 `site-packages`。

### Windows：命令提示符（CMD）

上面 `Set-Location` 一行换为 `cd /d ".\robotwin-triworld-train"`，PowerShell 的条件创建环境命令换为：

```bat
if not exist ".venv" py -3 -m venv .venv
```

其余单行 `git` 和 Python 命令可照常使用。下文 Windows 数据命令都写成单行并使用双引号，可以在 PowerShell 或 CMD 中运行。两种终端的条件语法不同，不要混用。

### Linux

```bash
git clone https://github.com/FKEY1520/robotwin-triworld-train.git robotwin-triworld-train
cd robotwin-triworld-train
python3 --version
if [ ! -e .venv ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import sys, numpy, h5py, PIL; print(sys.version); print('Dependencies OK')"
.venv/bin/python convert_dataset.py --help
```

所有示例都直接调用虚拟环境中的 Python，**不需要激活环境，也不需要更改 PowerShell 执行策略**。以后使用时进入项目目录，直接执行数据命令，不必每次创建环境。

可选安装方式：Windows 执行 `.\.venv\Scripts\python.exe -m pip install -e .`，Linux 执行 `.venv/bin/python -m pip install -e .`。安装后该环境中的 `triworld-train` 入口与 `python convert_dataset.py` 功能相同。

<a name="source-data"></a>

## 3. 准备已经转换好的任务包

建议数据与本项目放在并列目录。下面所有通用示例均从 `robotwin-triworld-train/` 项目根目录执行：

```text
your_workspace/
├── robotwin-triworld-train/                 # 本项目，命令执行位置
│   ├── convert_dataset.py
│   ├── configs/conversion48.json
│   ├── triworld_train/
│   ├── .venv/                              # 本机安装时创建
│   └── datasets/                           # 本机构建输出，不随 Git 克隆
└── robotwin-to-triworldbench/               # 上一步格式转换项目
    └── outputs/                            # --source-root 指向这里
        └── adjust_bottle_clean50/           # --tasks 使用这个目录名
            ├── robotwin_episode_mapping.json
            ├── test_dataset/
            │   ├── data/episode1.hdf5
            │   └── instructions/episode1.json
            ├── lossless/episode1/           # 若上一步生成了无损 PNG
            │   ├── head/frames/frame_00000.png
            │   ├── left/frames/frame_00000.png
            │   └── right/frames/frame_00000.png
            └── gt_dataset/episode1/         # 上一步生成的 JPEG 序列
                ├── head/frames/frame_00000.jpg
                ├── left/frames/frame_00000.jpg
                └── right/frames/frame_00000.jpg
```

上图只展开一条轨迹和每个视角的首张图片；实际需要映射中所有选中轨迹的全部动作、指令和连续图片。保留上一步的完整任务包，不要只复制 `test_dataset/first_frame/`。其他上游文件可以保留，本工具按所需路径读取。

`--source-root` 可以指向含多个任务包的父目录，也可以直接指向一个含 `robotwin_episode_mapping.json` 的任务包。`convert` 和普通 `inspect` 只识别源根及其直接子包；`batch` 默认递归，`inspect --recursive` 可预检查同样的目录树。`--tasks` 选择转换包，不是内部任务标签 `adjust_bottle`；递归扫描存在同名目录时应使用源根下的相对路径，例如 `outputs/adjust_bottle_clean50`。

指令文件需要是已选定的 JSON 字符串，或 `{"instruction":"..."}`，并与映射中的已选指令一致。本工具沿用上一步选择的句子，不会重新从 `seen` / `unseen` 候选中选择。

默认 `--image-source auto` 按每条轨迹选择：三视角 PNG 文件序列完整时使用 PNG，否则尝试完整 JPEG。一次轨迹只采用一种编码，不按缺帧位置混用。两种序列都不完整会失败；完整 PNG 中存在损坏图片时，正式构建会报错，不会静默换成 JPEG。需要强制选择时可指定 `png` 或 `jpg`。

<a name="first-build"></a>

## 4. 完成第一次训练集构建

以下使用通用相邻目录布局。首次试构建输出为 `datasets/adjust_bottle_smoke48`，正式输出为 `datasets/adjust_bottle_train48_v1`；两者独立，不能使用同一个已存在的目录。

### 4.1 确认当前目录和入口

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py --help
```

帮助中应列出 `inspect`、`convert`、`batch`、`verify`、`export-npz`。如果无法找到脚本，先回到含 `convert_dataset.py` 的目录。

### 4.2 预检查全部选中轨迹

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py inspect --source-root "../robotwin-to-triworldbench/outputs" --tasks adjust_bottle_clean50 --config "./configs/conversion48.json" --limit 0
```

`inspect` 读取映射、指令和动作，检查图片文件名及数量，计算计划窗口数，并列出可用数值字段。`--limit 0` 表示每个选中包的全部轨迹；`--limit 1` 表示按编号排序后的第一条，不是指定名为 `episode1` 的文件。

预检查不创建输出，也不解码所有图片，结果中的 `image_decode_verified` 为 `false`。它通过后仍需实际试构建，才能检查图片解码、RGB 模式及分辨率。

### 4.3 实际试构建一条轨迹

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py convert --source-root "../robotwin-to-triworldbench/outputs" --tasks adjust_bottle_clean50 --output-root "./datasets/adjust_bottle_smoke48" --config "./configs/conversion48.json" --limit 1
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py verify --dataset-root "./datasets/adjust_bottle_smoke48" --deep
```

这一步会实际读取整条轨迹、逐张解码检查图片、写入 episode 归档和窗口索引。完成后查看 `report.json`，深度校验末尾应包含 `"passed": true`。一个 episode 通常产生多个窗口，所以 `episode_count: 1` 不意味着 `window_count: 1`。

### 4.4 用新目录全量构建并校验

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py convert --source-root "../robotwin-to-triworldbench/outputs" --tasks adjust_bottle_clean50 --output-root "./datasets/adjust_bottle_train48_v1" --config "./configs/conversion48.json" --limit 0
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py verify --dataset-root "./datasets/adjust_bottle_train48_v1" --deep --report "./reports/adjust_bottle_train48_v1_verify.json"
```

这里的 `convert` 全量构建重新包含试构建的那条轨迹；它不只处理“剩余数据”。所有选中轨迹都必须成功，任意一条失败会停止本批原子构建，`convert` 没有追加或续转模式。输出先写入旁边的 `.partial-...` 临时目录，全部完成后才发布为正式目录。需要按任务保留进度时使用[批量模式](#batch-build)。

输出目录必须不存在，且不能放在源目录内部或作为源目录的父目录。再次运行请使用新的版本名，例如 `adjust_bottle_train48_v2`；不要预先手动创建最终输出目录。

Linux 对应命令仅需更换 Python 和脚本前缀，例如：

```bash
.venv/bin/python convert_dataset.py convert --source-root "../robotwin-to-triworldbench/outputs" --tasks adjust_bottle_clean50 --output-root "./datasets/adjust_bottle_train48_v1" --config "./configs/conversion48.json" --limit 0
.venv/bin/python convert_dataset.py verify --dataset-root "./datasets/adjust_bottle_train48_v1" --deep
```

### 4.5 本机 A 盘路径示例

如果代码和源数据确实位于以下位置，在 PowerShell 中使用绝对路径即可，不需要移动数据：

```powershell
Set-Location -LiteralPath "A:\Triworldbench训练数据集"
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py inspect --source-root "A:\robotwin_to_Triworldbench_github_ALOHA\robotwin-to-triworldbench\outputs" --tasks adjust_bottle_clean50 --config "./configs/conversion48.json" --limit 0
```

正式构建时沿用同一绝对 `--source-root`，输出仍选择新的目录。这里是已有本地环境的示例，克隆仓库不会自动得到这些路径、源数据或已生成训练集。

要递归扫描这个转换项目的所有任务，源根改为项目本身：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py batch --source-root "A:\robotwin_to_Triworldbench_github_ALOHA\robotwin-to-triworldbench" --output-root "./datasets/all_tasks_train48" --config "./configs/conversion48.json" --limit 0
```

首次输出须不存在；继续同一个批次时给这条命令追加 `--resume`。不要同时启动两次相同输出目录的构建。

<a name="window-rules"></a>

## 5. 理解历史、未来和末尾窗口

帧号从 0 开始。设未来起点为 `s`，图像目标为连续真实帧 `[s, s+H)`，即 `s` 到 `s+H-1`。合法范围为 `1 ≤ s ≤ T-H`，帧 0 用于历史。

```python
starts = list(range(1, T - H + 1, S))
if starts[-1] != T - H:
    starts.append(T - H)
```

默认 `H=48, S=8`。例如 `T=140`，起点为 `1, 9, 17, ..., 89, 92`，共 **13 个窗口**。最后一个起点 92 是额外补入的完整末窗，它的目标为 `92...139`，恰好覆盖视频最后 48 帧。

每条轨迹的未来帧 1 至末帧都会至少进入一个目标窗口；不跨轨迹拼接，也不在末尾重复图片伪造目标。相邻窗口的目标可以重叠，最后一个起点的间隔可以小于 `S`。减小步长会增加相关窗口，独立轨迹数量不会因此增加。

当 `P≥5` 时，历史由全局帧 0 和目标前最近 `P-1` 帧组成。默认 `P=9` 的例子：

| 未来起点 | 历史索引 | 未来图像目标 |
| --- | --- | --- |
| `s=1` | `[0,0,0,0,0,0,0,0,0]` | `1...48` |
| `s=3` | `[0,0,0,0,0,0,0,1,2]` | `3...50` |
| `s=20` | `[0,12,13,14,15,16,17,18,19]` | `20...67` |

历史不足时只在历史部分重复帧 0；图像和动作采用同样的索引。`P=1` 是单帧历史模式，取最近过去帧 `[s-1]`。因此最小轨迹长度为 `H+1`，不要求 `P+H`。

**`H=48` 在本阶段生成一个完整的 48 帧目标，不拆成三个 16 帧目标。** 后续训练代码须按相同 `P/H` 配置读取；如果下游重新切段，实际训练长度会改变。数据构建通过不能单独证明已经完成一次预测 48 帧的训练。

没有可靠采集 FPS 时，元数据保持 `fps: null`，时间依据为 `frame_index`。48 帧不自动等于 3 秒，播放视频的 FPS 也不能代替采集频率。

<a name="output-data"></a>

## 6. 看懂输出与 Python 读取

正式构建示例输出：

```text
datasets/adjust_bottle_train48_v1/
├── dataset.json                         # 汇总清单、参数、episode 记录
├── windows.jsonl                        # 全部训练窗口，每行一个索引
├── normalizer.json                      # 汇总训练动作统计
├── report.json                          # 数量、大小和构建结果
└── adjust_bottle_clean50/                # 此任务可独立使用的子数据集
    ├── dataset.json
    ├── windows.jsonl
    ├── normalizer.json
    ├── report.json
    └── episodes/
        ├── episode1.h5
        └── ...
```

每个选定 episode 的图片编码字节和完整动作归档一次，重叠窗口保存索引，不为每个窗口重复复制图片。程序不会跨重复转换包自动去重。PNG/JPEG 字节保留在 `/frames/head`、`/frames/left`、`/frames/right`；标准动作为 `/joint_action/vector`，源 HDF5 的全部数值字段保留在 `/source_numeric/...`。

归档不缩放图片。默认字段为 float64 时，保存的动作保持 float64；读取训练样本时才转为 float32。采用插件或其他动作字段时，实际类型及变换以清单记录为准。JPEG 本身已有的有损压缩不会因为归档变成无损图片。

每条轨迹实际采用的编码记录在 `dataset.json` 的 `episodes[*].provenance.image_encoding` 中；`inspect` 提供 `image_encodings` 汇总。`report.json` 的 `settings.image_source` 记录的是选择配置，不是逐轨迹实际编码。

从项目根目录运行 Python，或先在所用环境中 `pip install -e .`，再使用：

```python
from triworld_train.dataset import WindowDataset

dataset = WindowDataset("datasets/adjust_bottle_train48_v1", image_size=(256, 320))
sample = dataset[0]
print(len(dataset))
print(sample["history_video"].shape)   # (3, 3, 9, 256, 320)
print(sample["future_video"].shape)    # (3, 3, 48, 256, 320)
print(sample["packed_actions"].shape)  # (57, 14)
print(sample["prompt"])
```

| 返回字段 | 通用形状 / 含义 |
| --- | --- |
| `history_video` | `[3, 3, P, h, w]`：视角、RGB 通道、历史时间、高、宽。 |
| `future_video` | `[3, 3, H, h, w]`：相同视角顺序的未来图像目标。 |
| `packed_actions` | 默认 `[P+H, 14]`，先历史后未来，与图片索引对齐的已知动作条件；改变动作选择时列数为所配维数。 |
| `prompt` | 该 episode 已选择的任务指令。 |
| `metadata` | 任务、数据角色、视角顺序和时间依据等信息。 |

视角顺序固定为 `head, left, right`。图像返回 float32、范围 `[-1,1]`；`image_size=(h,w)` 只在读取时缩放，不修改磁盘归档。省略尺寸时使用源分辨率。`packed_actions` 返回原单位的 float32 数值，读取器**不自动应用动作归一化**。在已安装 PyTorch 的训练环境中可选 `as_torch=True`。

`normalizer.json` 按各条完整训练轨迹计算逐维第 1/99 百分位，每个时间步只统计一次，重叠窗口和历史补帧不重复加权。训练入口应使用配套统计及一致的归一化方法，避免重复归一化。完整字段说明见 [数据格式](docs/DATA_FORMAT.md)。

完整复制汇总目录及其子目录，即可搬运全部任务；单独复制一个完整任务子目录也可读取。路径引用是相对路径，图片已经归档，搬到其他机器后不要求原始源目录存在。移动后再运行 `verify --deep`。

### 批量模式的目录与报告

`batch` 的任务子目录使用稳定 ID。源根的直属包通常保留目录名；更深层的包使用 `包名--相对路径摘要`，即使后来出现同名包，也不改变已有任务 ID。任务选择仍使用源相对路径，而非猜测输出 ID。

```text
datasets/all_tasks_train48/
├── batch_state.json                     # 续跑状态，保留且不要手改
├── batch_report.json                    # 每次批量运行的结果与错误
├── dataset.json                         # 本次所选任务全部成功后发布
├── windows.jsonl
├── normalizer.json
├── report.json
└── <稳定任务ID>/
    ├── .batch_source.json               # 续跑来源校验记录，完整复制时保留
    ├── dataset.json
    ├── windows.jsonl
    ├── normalizer.json
    ├── report.json
    └── episodes/*.h5
```

`batch_report.json` 中 `aggregate_ready: true` 表示本次汇总入口已就绪；`completed_count` 包含新完成及复用成功的任务，`converted_count` 是本次新构建数，`skipped_count` 是校验后复用数，`failed_count` 是失败数，`tasks` 记录各包状态与 `error`。存在失败时根汇总不发布，已完成任务子目录仍可单独校验和读取。若包全部完成但汇总出错，查看 `aggregate_error`。`--limit 1` 也可以得到完整的调试批次，但它只包含每个包的一条轨迹，不是全量数据。

<a name="multiple-tasks"></a>

## 7. 新增任务与多任务训练

上一步转换了新任务后，将完整新包保留在 `outputs/` 同级：

```text
robotwin-to-triworldbench/outputs/
├── adjust_bottle_clean50/
├── beat_block_hammer_clean50/
└── beat_block_hammer_randomized500/
```

只有实际存在第二个包时，才将下面的包名用于命令。先检查，再生成新任务的独立输出：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py inspect --source-root "../robotwin-to-triworldbench/outputs" --tasks beat_block_hammer_clean50 --config "./configs/conversion48.json" --limit 0
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py convert --source-root "../robotwin-to-triworldbench/outputs" --tasks beat_block_hammer_clean50 --output-root "./datasets/beat_block_hammer_train48_v1" --config "./configs/conversion48.json" --limit 0
```

联合训练可以明确列出正式包；递归扫描用相对于源项目根目录的路径：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py inspect --source-root "../robotwin-to-triworldbench" --recursive --tasks outputs/adjust_bottle_clean50 outputs/beat_block_hammer_clean50 --config "./configs/conversion48.json" --limit 0
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py batch --source-root "../robotwin-to-triworldbench" --tasks outputs/adjust_bottle_clean50 outputs/beat_block_hammer_clean50 --output-root "./datasets/selected_tasks_train48_v1" --config "./configs/conversion48.json" --limit 0
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py verify --dataset-root "./datasets/selected_tasks_train48_v1" --deep
```

`--tasks` 可以用唯一的包名简写，但多处同名时必须给相对路径，程序会列出可选路径。目录名不同的重复数据、smoke 和旧版包不会自动去重；全项目自动扫描也会发现这些有效包。联合训练前用 `inspect --recursive` 核对列表，选择本次所需正式包，避免同一轨迹多次参与训练和统计。

联合输出的根清单包含所有选中任务，各任务子目录仍然独立。联合训练读取根目录，单任务训练读取对应子目录。不要同时把根目录、它的任务子目录、旧版本和 smoke 输出一起加入训练，否则会重复使用同一批轨迹。

后来把新包放入同一个源项目后，要更新第 0 节的批量输出，可再次运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py batch --source-root "../robotwin-to-triworldbench" --output-root "./datasets/all_tasks_train48" --config "./configs/conversion48.json" --limit 0 --resume
```

它重新发现本次源包，校验后复用未变的完成任务，构建新增或此前未完成的包，再重建本次所选任务的汇总。若最初使用了 `--tasks`，更新时也要明确选择所需的新旧包。新增任务处理期间旧根汇总暂时撤下，只有本次全部成功后才再次发布；以 `batch_report.json` 的 `aggregate_ready` 为准，不把上次的成功报告当成本次完成。

本工具没有目录监视或后台自动更新功能。放入新包后需要手动运行；`convert` 仍要求新的输出目录，只有 `batch --resume` 可以继续本工具已有的批量输出。已完成源包或配置改变时应使用新输出版本，不覆盖旧数据。

<a name="customization"></a>

## 8. 修改帧数、动作与数据内容

### 历史帧、未来帧和步长分别改哪里

这三个参数可以分别修改，无需改 Python 源码：

| 你想改变的内容 | JSON 中修改的键 | 命令行示例 | 其他时间参数保持默认时的结果 |
| --- | --- | --- | --- |
| 使用多少帧历史 | `history_frames` | `--history-frames 13` | 13 帧历史 + 48 帧未来；历史充足时为首帧锚点 + 最近 12 帧。 |
| 一次窗口包含多少帧未来目标 | `horizon` | `--horizon 32` | 9 帧历史 + 32 帧未来；轨迹至少需要 33 个真实时刻。 |
| 重叠采样的密度 | `stride` | `--stride 4` | 仍为 9 帧历史 + 48 帧未来，常规起点间隔缩短到 4 帧。 |

历史长度仍需为 `4n+1`，未来长度仍需为正的 4 的倍数；参数不是任意正整数。修改历史长度改变的是历史采样数量，不会自动切换为“最近连续 P 帧”的采样策略，第 5 节给出了具体索引。

例如只把历史改为 13 帧，保留未来 48 帧和步长 8：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py convert --source-root "../robotwin-to-triworldbench/outputs" --tasks adjust_bottle_clean50 --output-root "./datasets/adjust_bottle_train48_p13_v1" --config "./configs/conversion48.json" --horizon 48 --history-frames 13 --stride 8 --limit 0
```

### 配置 JSON 与命令行优先级

`configs/conversion48.json` 内容为：

```json
{
  "horizon": 48,
  "stride": 8,
  "history_frames": 9,
  "image_source": "auto",
  "action_keys": ["joint_action/vector"],
  "expected_action_dim": 14
}
```

JSON 只接受以上六个键。**显式命令行参数优先于 JSON**，未指定的参数使用 JSON 或内置默认值。这里 `--config` 指参数文件路径，与上一步格式转换工具中表示原始配置目录名的同名参数含义不同。

希望保存新方案时，先复制该 JSON 为新文件，例如 `configs/conversion24_p5.json`，将 `horizon` 改为 24、`history_frames` 改为 5，并在 `inspect` 和 `convert` 中都指定这个新文件。命令行覆盖只作用于本次运行，不会把新值写回 JSON；实际生效的参数可以在输出的 `report.json` 中查看。

例如改为未来 `H=24`、历史 `P=5`、步长 `S=8`，预检查和实际构建必须使用相同参数：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py inspect --source-root "../robotwin-to-triworldbench/outputs" --tasks adjust_bottle_clean50 --config "./configs/conversion48.json" --horizon 24 --history-frames 5 --stride 8 --limit 0
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py convert --source-root "../robotwin-to-triworldbench/outputs" --tasks adjust_bottle_clean50 --output-root "./datasets/adjust_bottle_train24_p5_v1" --config "./configs/conversion48.json" --horizon 24 --history-frames 5 --stride 8 --limit 0
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py verify --dataset-root "./datasets/adjust_bottle_train24_p5_v1" --deep
```

对应读取结果为历史 `[3,3,5,h,w]`、未来 `[3,3,24,h,w]`、动作 `[29,14]`。下游训练配置也须改为相同的历史和未来长度；只改窗口文件不会自动更改训练代码。

### 动作字段

先用 `inspect` 查看 `available_numeric_fields`，再通过 `--action-key <HDF5路径>` 选择。需要拼接多个时间对齐字段时，可重复这个参数；按给出顺序拼接，合计列数必须等于 `--action-dim`。JSON 中对应 `action_keys` 列表和 `expected_action_dim`。

显式提供 `--action-key` 会替代默认字段选择。拼接后统一保存到 `/joint_action/vector`，源 HDF5 的全部数值字段仍保留在 `/source_numeric`。改变字段或维数需要同步调整后续读取和训练约定，转换器不会推断关节顺序或动作物理含义。

### 自定义 Python 插件

`--transform "脚本路径:函数名"` 在每条轨迹归档前调用本地函数。示例接口：

```python
def customize_episode(episode):
    # actions: [T,D] NumPy 数组；prompt: 字符串
    # images: head/left/right 对应的有序图片路径列表；metadata: 字典
    episode.prompt = episode.prompt.strip()
    episode.metadata["custom_transform"] = {"name": "strip_prompt"}
    return episode
```

仓库已提供可运行示例：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py convert --source-root "../robotwin-to-triworldbench/outputs" --tasks adjust_bottle_clean50 --output-root "./datasets/adjust_bottle_train48_custom_v1" --config "./configs/conversion48.json" --transform "./examples/custom_transform.py:customize_episode" --limit 0
```

插件会执行 Python 代码，只使用你信任的本地脚本。它可以修改提示词、动作值或图片路径，但必须返回 `EpisodeData`，保持原帧数和所选动作维数。程序检查形状、数量和数值有效性，**插件作者负责保证三视角与动作的时间顺序和语义对齐**。裁帧或重采样应先生成新的源包。插件路径、摘要和修改元数据会记录在输出中。

要修改数据，优先调整源包、配置或插件，再生成新版本。直接编辑输出 HDF5、窗口索引或统计文件会破坏校验摘要；不要只改校验值来绕过验证。更多例子见 [修改数据与参数](docs/CUSTOMIZATION.md)。

<a name="cli-options"></a>

## 9. 完整命令行参数

使用 `python convert_dataset.py <子命令> --help` 查看当前版本帮助。路径相对于命令执行目录解析，含空格的路径使用双引号。

| 子命令 | 必填参数 | 功能 |
| --- | --- | --- |
| `inspect` | `--source-root` | 读取源包并规划窗口，不创建数据集。 |
| `convert` | `--source-root`、`--output-root` | 构建 episode 归档、窗口和动作统计。 |
| `batch` | `--source-root`、`--output-root` | 递归发现源包，逐包构建并记录可继续的进度。 |
| `verify` | `--dataset-root` | 验证已经生成的数据集。 |
| `export-npz` | `--dataset-root`、`--output-root` | 将窗口展开为便于检查的 NPZ 文件。 |

`inspect` / `convert` / `batch` 共用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--source-root` | 必填 | 一个任务包、包父目录；批量递归时可使用整个转换项目。 |
| `--tasks NAME [NAME ...]` | 全部可识别包 | 按包名或源相对路径选择；递归扫描中同名时用相对路径消歧。 |
| `--config FILE` | 无 | JSON 参数文件，仅接受第 8 节六个键。 |
| `--limit N` | `0` | **每个选中任务包**最多读取的 episode 数；0 为全部。 |
| `--horizon H` | `48` | 正的 4 的倍数；真实连续未来图像目标长度。 |
| `--history-frames P` | `9` | `4n+1`；历史输入长度，允许早期帧 0 填充。 |
| `--stride S` | `8` | 正整数，不能超过 `H`。 |
| `--image-source` | `auto` | `auto`、`png` 或 `jpg`，以整条 episode 为单位选择。 |
| `--action-key KEY` | `joint_action/vector` | 可重复，按顺序拼接；JSON 键为 `action_keys`。 |
| `--action-dim D` | `14` | 拼接后预期维数；JSON 键为 `expected_action_dim`。 |

`convert` 和 `batch` 接受 `--output-root`（输出目录）与 `--transform FILE:FUNCTION`（可选插件）。`inspect` 不接受这两个参数，也不会执行转换插件。`--recursive` 和 `--resume` 属于运行方式参数，不能写入六键配置 JSON。

| 扫描与继续参数 | 说明 |
| --- | --- |
| `inspect --recursive` | 递归预检查；默认 inspect 仍只扫描源根及直接子包，遇错立即停止。 |
| `batch` 默认扫描方式 | 递归；不需要额外 `--recursive`，默认 `--limit 0` 为每包全部轨迹。 |
| `batch --resume` | 明确继续已存在的受管理批量输出；验证源、数据配置及完成归档后跳过有效完成项。 |
| `convert --output-root DIR` | 必须不存在；不接受 `--resume`。 |
| `batch --output-root DIR` | 首次必须不存在，后续相同批量目录显式使用 `--resume`；不能接管任意旧目录。 |

| 其他子命令参数 | 说明 |
| --- | --- |
| `verify --dataset-root DIR` | 可指向汇总根目录或完整任务子目录。 |
| `verify --deep` | 在摘要、索引、动作和统计检查之外，解码全部归档图片。 |
| `verify --report FILE` | 可选，将校验结果另存为 JSON；建议放在 `reports/`。 |
| `export-npz --dataset-root DIR` | 要导出的已构建数据集。 |
| `export-npz --output-root DIR` | 不存在的新目录，必须在源数据集树之外。 |
| `export-npz --limit N` | 最多导出 N 个**窗口**，0 为全部，与构建时的按任务轨迹限制不同。 |
| `export-npz --height h --width w` | 可选，必须成对提供；只调整导出图片尺寸。 |

导出一个样本供检查：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_dataset.py export-npz --dataset-root "./datasets/adjust_bottle_train48_v1" --output-root "./exports/adjust_bottle_sample48_v1" --limit 1 --height 256 --width 320
```

NPZ 包含 `history_video`、`future_video`、`packed_actions` 和 `metadata_json`，可用 `numpy.load(path, allow_pickle=False)` 读取。它会展开重叠窗口的图片，完整导出可能远大于 episode 归档；通常先导出一个窗口即可。

<a name="troubleshooting"></a>

## 10. 常见问题与排查

| 现象 / 问题 | 检查与处理 |
| --- | --- |
| 找不到任务或 `Unknown bundle names` | 检查根目录及 `--tasks` 包名，确认该目录含 `robotwin_episode_mapping.json`；原始 Raw 目录不能直接使用。 |
| `No complete synchronized ... image set` | 检查三视角、编号、数量和扩展名。需要连续的全部图片；可先检查上一步转换报告。 |
| 预检查通过，正式构建图片报错 | `inspect` 不解码所有图片；检查错误帧是否损坏、非 RGB 或尺寸不一致。 |
| `Need at least ... real frames` | 默认每条至少 49 帧。可选符合约束的较短 `H`，并同步训练配置；不能补造未来目标。 |
| 动作维数不符 | 对照 `inspect` 字段与列数，核对动作选择和 `--action-dim`，不要仅改维数参数掩盖不同动作含义。 |
| `Refusing to overwrite ...` | `convert` 使用新的版本目录；已有受管理批量目录用相同 batch 参数加 `--resume`，不能拿它覆盖任意旧输出。 |
| SHA-256 或窗口一致性校验失败 | 检查传输完整性及是否手动改过文件，优先从可信源恢复或用原配置重新构建。 |
| 改了窗口参数但数量没有变化 | 确认读的是本次新输出，查看 `report.json` 中 `settings`；重跑 inspect 时也要提供相同参数。 |
| 批量报告里有失败，其他包是否白跑 | 不会。完成包已保留，先查看 `tasks[*].error`，修复未完成源包后用同一命令加 `--resume`。 |
| resume 提示已完成源包变化或归档损坏 | 不会静默重建或覆盖；恢复原文件，或使用新输出版本完整构建。改变数据配置也使用新版本。 |
| batch 有任务子目录但根清单不存在 | 本次仍有未完成/失败任务，查看 `batch_report.json`；全部成功后才发布联合训练入口。 |
| 是否自动处理后来下载的数据 | 不会后台监视。先完成上一步格式转换，再按第 7 节手动 `batch --resume` 或使用新输出版本。 |
| 是否可以直接开始模型训练 | 本仓库完成数据构建和读取。训练入口需要按相同数据约定接入，模型代码后续单独发布。 |

命令出现 `ERROR:` 或非零退出码时应停止后续步骤，先处理原因。已捕获的构建错误会清理本次临时目录；强制中止或断电可能留下 `.partial-...`，它不是可训练的完整输出。

<a name="verification"></a>

## 11. 测试、验证记录与适用范围

在现有虚拟环境中安装测试依赖并运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Linux 使用 `.venv/bin/python` 执行相同模块命令。测试覆盖窗口边界、历史与动作对齐、末窗、图片来源选择、数据搬运、统计、插件、NPZ 和损坏文件拒绝；GitHub Actions 配置覆盖 Python 3.10 / 3.12。

历史验证样例 `adjust_bottle_clean50` 包含 50 条轨迹、7,238 个同步时刻，在 `P=9,H=48,S=8` 下生成 664 个窗口，归档 21,714 张 PNG。它是**过去在本机运行的示例记录**，不代表所有任务都得到这些数量，数据也没有打包在 GitHub 克隆中。详见 [验证记录](docs/VERIFICATION.md)。

对自己的任务仍应执行完整的 `verify --deep`，并让实际训练读取器读一个样本。CPU 数据转换验证与 GPU 编码、训练更新和训练质量分别验证，不能相互替代。

相关文档：[数据格式](docs/DATA_FORMAT.md)、[自定义接口](docs/CUSTOMIZATION.md)、[GitHub 上传与数据交付](docs/GITHUB.md)。已有训练环境的缓存适配另见 [部署接入说明](docs/LRZ.md)；该部分需要另行提供匹配的模型代码与运行环境。

<a name="update-project"></a>

## 12. 更新代码并保留本地数据

使用 Git 克隆的项目，可在仓库根目录检查后更新：

```powershell
git status --short
git ls-files -- datasets exports cache lrz_cache reports .venv
git pull --ff-only
```

先检查前两条输出。当前 `.gitignore` 排除了 `datasets/`、`exports/`、缓存、报告、虚拟环境和常见大数据文件。只要本地和远端都没有跟踪这些路径，正常拉取会保留本地输出，不会自动重新构建。**忽略规则不能保护已经被 Git 跟踪的文件**；有本地代码修改时先保留并处理，再更新。

`--ff-only` 在提交分叉时会停止，不要为更新文档直接覆盖本地修改。保留原来的 `.venv`，仅在依赖要求变化时使用它重新安装 `requirements.txt`。不要用 `git clean -fdx` 清理项目，它会删除被忽略的数据和环境。

从 ZIP 获取的目录没有 Git 历史，不能直接 `git pull`。将新版解压到另一目录，再迁移所需代码和文档，保留已有数据、环境及本地修改。代码更新与数据版本更新分开进行；更改数据定义后，仍使用新输出目录构建并校验。
