# 修改转换参数与数据接口

修改数据后使用新的输出目录，并重新运行 `verify --deep`。原始 `outputs` 数据保留为可重复构建的来源。

## 参数文件

默认文件为 `configs/conversion48.json`：

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

| 配置键 | 命令行参数 | 调整效果 |
| --- | --- | --- |
| `horizon` | `--horizon` | 未来连续目标帧数 |
| `stride` | `--stride` | 相邻常规窗口的起点间隔 |
| `history_frames` | `--history-frames` | 包含锚点帧的历史长度 |
| `image_source` | `--image-source` | `auto`、`png` 或 `jpg` |
| `action_keys` | 重复 `--action-key` | 从 HDF5 选择并按顺序拼接动作字段 |
| `expected_action_dim` | `--action-dim` | 最终动作维数检查 |

约束来自当前 Wan 视频模型：`horizon` 必须是正的 4 的倍数；`history_frames` 必须为 `4n+1`（1、5、9、……）；`stride` 为正整数且不能大于 `horizon`。episode 必须至少有 `horizon+1` 个真实帧。显式命令行参数优先于配置 JSON 中的默认值。

例如保留 48 帧未来、提高窗口密度：

```powershell
.\.venv\Scripts\python.exe .\convert_dataset.py convert `
  --source-root 'A:\robotwin_to_Triworldbench_github_ALOHA\robotwin-to-triworldbench\outputs' `
  --output-root '.\datasets\train48_stride4' `
  --tasks adjust_bottle_clean50 `
  --horizon 48 --stride 4 --history-frames 9
```

步长越小，训练索引越多，同一帧被不同上下文使用的次数越多。转换器拒绝步长大于 `horizon` 的设置，以保持目标帧连续覆盖。默认 8 可以覆盖所有可用未来帧。

改变 `horizon` 或 `history_frames` 后，须对应调整模型的视频长度和已有 LRZ 训练配置。默认 9 + 48 = 57 的打包约定见 [LRZ 接入](LRZ.md)。改变动作维数也需要调整模型，而非仅通过数据转换改变训练网络。

## 选择其他动作字段

字段路径是源 HDF5 中的路径。使用多个 `--action-key` 时按给出的顺序拼接：

```powershell
# 将下面的路径替换为 inspect 中实际存在且时间对齐的数值字段。
.\.venv\Scripts\python.exe .\convert_dataset.py convert `
  --source-root 'A:\robotwin_to_Triworldbench_github_ALOHA\robotwin-to-triworldbench\outputs' `
  --output-root '.\datasets\custom_actions' `
  --tasks adjust_bottle_clean50 `
  --action-key 'joint_action/vector' --action-dim 14 `
  --horizon 48 --stride 8 --history-frames 9
```

拼接后的动作仍保存到标准归档路径 `/joint_action/vector`，这样训练读取接口保持一致。源数值字段同时保留在 `/source_numeric`。不要把不同物理含义、不同时间采样的字段直接拼接为同一动作，先在插件中完成所需的对齐和转换。

## Python 插件接口

命令行用 `--transform <脚本路径>:<函数名>` 加载函数。每个 episode 在归档及窗口构建前调用一次，函数接收并返回 `EpisodeData`：

```python
# EpisodeData 公共字段：
# actions: np.ndarray，形状 [T, D]
# prompt: str
# images: dict[str, list[pathlib.Path]]，head/left/right 的逐帧路径
# metadata: dict，扩展和追溯信息

def customize_episode(episode):
    episode.prompt = episode.prompt.strip()
    episode.metadata["custom_transform"] = {
        "name": "customize_episode",
        "description": "Strip whitespace from prompt"
    }
    return episode
```

已提供可运行示例 `examples/custom_transform.py`：

```powershell
.\.venv\Scripts\python.exe .\convert_dataset.py convert `
  --source-root 'A:\robotwin_to_Triworldbench_github_ALOHA\robotwin-to-triworldbench\outputs' `
  --output-root '.\datasets\train48_custom' `
  --tasks adjust_bottle_clean50 `
  --config '.\configs\conversion48.json' `
  --transform '.\examples\custom_transform.py:customize_episode'
```

插件可以修改提示词、动作数值、同一维数内的列顺序及图片路径。返回后必须保持源 episode 的帧数，以及三视角图片和动作的时间对齐。当前接口不支持在插件中裁去帧或重采样，因为归档同时保存原始数值通道；需要这种操作时，先构建时间对齐的新源任务包，再交给转换器。动作维数由源 `action_keys` 的选择和 `--action-dim` 控制，插件应保持该维数。

动作单位变换也可在插件中实现。例如只有在源定义确认为毫米、模型定义确认为米时，才将相关列除以 1000，并记录修改列及变换：

```python
def convert_selected_units(episode):
    episode.actions = episode.actions.astype("float64", copy=True)
    columns = [0, 1, 2]  # 替换为你确认需要单位转换的列
    episode.actions[:, columns] /= 1000.0
    episode.metadata["custom_transform"] = {
        "name": "convert_selected_units",
        "columns": columns,
        "from_unit": "mm",
        "to_unit": "m"
    }
    return episode
```

保存插件代码及它的版本，便于解释训练结果。构建后归一化统计会基于修改后的动作重新计算。

## 图片尺寸与图像修改

归档保留源图片编码和分辨率。模型需要的尺寸在训练读取时设置：

```python
dataset = WindowDataset("datasets/train48", image_size=(256, 320))
```

如果需要裁剪、遮挡或其他永久图像处理，可在插件中生成新图片，然后更新 `episode.images` 指向这些文件；三个视角仍须具有相同的帧数和时间对齐。修改保存的新图片格式后，实际 RGB 编码会随归档一起打包。

## 数据版本记录

一次训练建议同时保存：代码 commit、配置 JSON、插件文件、`dataset.json`、`normalizer.json`、构建 `report.json` 和 LRZ 训练配置。`--limit` 生成的是小规模调试数据，不应和正式完整训练集共用输出版本。
