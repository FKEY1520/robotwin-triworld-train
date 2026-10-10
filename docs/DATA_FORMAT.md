# 训练数据格式与窗口语义

## 输入数据

源根目录包含多个任务包。已识别的 RoboTwin → TriWorldBench 输出结构如下：

```text
outputs/<任务包>/
├── robotwin_episode_mapping.json
├── test_dataset/
│   ├── data/episodeN.hdf5
│   └── instructions/episodeN.json
├── lossless/episodeN/
│   ├── head/frames/frame_00000.png
│   ├── left/frames/frame_00000.png
│   └── right/frames/frame_00000.png
└── gt_dataset/episodeN/
    ├── head/frames/frame_00000.jpg
    ├── left/frames/frame_00000.jpg
    └── right/frames/frame_00000.jpg
```

图像帧与动作行按 frame index 对齐。视角固定为 `head, left, right`；对应 RoboTwin 的 `observation/head_camera/rgb`、`observation/left_camera/rgb`、`observation/right_camera/rgb`。不要把左右视角改名为腕部相机，除非新源数据的实际相机定义发生了改变。

默认动作来自 `joint_action/vector`，维数为 14。原始动作和其余可保存的数值字段保留在归档中，转换插件处理后的动作写入标准动作位置，供训练读取。

`image_source=auto` 对每个 episode 优先使用完整 PNG，否则使用完整 JPEG。完整是指三视角均拥有与动作行数匹配、从帧 0 连续排列的图片。缺帧、维数错误或其他对齐问题不应被静默补为未来目标。

## 输出目录

每个任务目录包含下列文件；父级汇总目录也包含 JSON/JSONL 元数据，引用全部任务：

| 文件 | 用途 |
| --- | --- |
| `dataset.json` | 数据集版本、转换参数、视角与其他读取配置 |
| `windows.jsonl` | 每行一个训练窗口，使用相对路径引用 episode 归档 |
| `normalizer.json` | 本训练集的逐动作维度归一化统计 |
| `report.json` | 构建结果和来源信息 |
| `episodes/*.h5` | 每个 episode 的完整图片、动作及元数据 |

`windows.jsonl` 每行包含 `sample_id`、`episode_id`、`source_uid`、`split`、`bundle`、`task`、`future_start`、`horizon`、`history_indices` 和 `target_end_exclusive`。其中目标区间为 `[future_start, target_end_exclusive)`，`split` 恒为 `train`。`dataset.json` 中的 episode 记录通过 `episode_id` 提供对应的归档相对路径、提示词、帧数、图像尺寸及来源。

episode HDF5 的主要路径：

```text
/joint_action/vector  # 标准动作，[T, D]，保持转换后原始数值精度
/frames/head          # 按时间排列的编码图片字节
/frames/left
/frames/right
/source_numeric/...   # 源文件的其他数值数组
```

图片以原来的 PNG/JPEG 编码字节存入 HDF5 可变长字节数组，读取时才解码为 RGB。归档内部不需要源文件继续存在；元数据中保存的原始路径用于追溯。移动数据时完整复制汇总目录及其子目录，或只复制一个完整任务目录。

## 48 帧未来窗口

用 `T` 表示一个 episode 的总帧数，使用从 0 开始的帧号。未来长度 `H=48`，步长 `S=8`：

1. 合法未来起点为 `1 ≤ s ≤ T-H`，帧 0 保留为初始历史。
2. 常规起点为 `1, 1+S, 1+2S, ...`，不超过 `T-H`。
3. 如果最后常规起点不是 `T-H`，再追加起点 `T-H`，并去重。
4. 每个窗口目标为 `[s, s+1, ..., s+H-1]`，均来自真实帧。

因此默认配置要求 episode 至少有 49 帧才能形成一个窗口。最后一个目标一定是视频最后 48 帧 `[T-48, ..., T-1]`。目标不做尾部填充；窗口不会跨 episode 或任务拼接。

例如 `T=65`：

```text
未来起点：  1, 9, 17
最后目标： 17, 18, ..., 64
```

例如 `T=70`：

```text
常规起点：  1, 9, 17
额外末窗： 22
最后目标： 22, 23, ..., 69
```

默认步长 8 小于目标长度 48，目标窗口相互重叠，同时覆盖帧 1 至最后一帧。减小步长会增加相关训练样本，但不会增加独立 episode 数量。

## 9 帧历史与 57 帧打包

为匹配当前模型输入，历史由 **初始锚点帧 0** 与 **紧接目标之前的最近 8 帧** 组成。对未来起点 `s`：

```python
history_indices = [0] + [max(0, s - 8 + i) for i in range(8)]
future_indices = list(range(s, s + 48))
packed_indices = history_indices + future_indices
```

早期窗口历史不足时重复帧 0，保证输入长度固定；未来 48 帧始终是真实连续帧。动作使用同一组索引，所以历史补帧也对应帧 0 的动作。9 帧历史 + 48 帧未来形成 57 帧视频/动作序列，与 LRZ 模型适配约定一致。

## Python 读取结果

`WindowDataset` 使用 `(height, width)` 配置加载时的图片尺寸；默认模型示例使用 `(256, 320)`。

| 字段 | 形状 / 类型 | 含义 |
| --- | --- | --- |
| `history_video` | `[3, 3, 9, H, W]`, float32 | 视角、RGB 通道、时间、高、宽 |
| `future_video` | `[3, 3, 48, H, W]`, float32 | 同样顺序的未来目标 |
| `packed_actions` | `[57, 14]`, float32 | 与打包视频时间索引对齐的原始单位动作，尚未归一化 |
| `prompt` | str | 本 episode 的任务提示词 |
| metadata | 读取器提供的追溯信息 | 用于核对任务、episode 和窗口来源 |

RGB 图像归一化为 `pixel / 127.5 - 1`，取值范围为 `[-1, 1]`。`as_torch=True` 可返回训练环境使用的张量。精确返回字段以读取器实现为准，以上列出模型接入所需的稳定字段。

FPS 未经来源确认时保存为 `null`；窗口的连续性基于 `frame_index`。48 帧并不自动代表 3 秒或其他固定时长。

## 动作归一化

统计在唯一 episode 动作上计算一次，每个源动作行只贡献一次；窗口重叠不会重复加权。全部统计来自当前训练集，没有访问现有验证集。

`normalizer.json` 的核心字段：

```json
{
  "schema": 1,
  "method": "bwm_p01_p99_clip",
  "q01": ["每个动作维度的第 1 百分位"],
  "q99": ["每个动作维度的第 99 百分位"],
  "eps": 1e-8,
  "constant_policy": "zero",
  "statistics": {
    "unique_frames": 0,
    "episode_count": 0,
    "overlap_weighted": false
  }
}
```

上面的数组和计数是格式说明，实际文件中为数字。对非常量维度 `d`：

```text
normalized[d] = clip(2 * (x[d] - q01[d]) / (q99[d] - q01[d] + eps) - 1, -1, 1)
```

范围小于或等于 epsilon 的常量维度输出 0。归档中的动作保留源精度（当前源为 float64），`WindowDataset` 只将它转为 float32，不自动归一化。现有 LRZ 训练入口使用原模型的同一百分位规则拟合缓存统计，在训练读取缓存时应用归一化；详见 [LRZ 接入](LRZ.md)。自行编写其他训练入口时应明确何处应用统计，避免归一化两次。跨任务训练时使用汇总训练集的统计；单任务训练可使用该任务自己的统计。数据版本和统计文件应一起记录，避免相同动作在不同训练阶段采用不同变换。
