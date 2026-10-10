# 在 LRZ 使用现有自研模型 1.0 训练

本项目只整理训练数据和对接缓存，不修改已有模型。已核对本机 `A:\Triworldbench测评模型自主开发树\LRZ上的测试\TriViewWan_LRZ.py` 的部署包，以及 `Wan2.2-TI2V-5B基座_Text_3view_action原生设计` 项目。其结构为原生 Wan2.2-TI2V-5B，14 维 `joint_action/vector`，每五个主干块插入跨视角模块（六处），内部宽度 256、8 个头、K/V 空间池化 4。视角顺序固定为 `head,left,right`。

每个训练样本包含 **9 帧历史 + 48 帧真实未来**。历史不足九帧时重复全局首帧；历史充分时选择原轨迹第零帧锚点和最近八帧。三个相机和动作使用相同的绝对索引。原训练器的 `schema=2` 缓存、历史 latent、流匹配损失与动作编码均保持一致。57 帧满足 VAE 的 `4n+1` 规则；48 帧未来对应十二个预测 latent，九帧历史对应三个条件 latent。

原 `python -m triviwan.cache` 只按不重叠窗口切分完整 episode。本项目的 `dataset.json` 和 `windows.jsonl` 应使用 **`scripts/prepare_lrz_cache.py`** 编码，再交给原 `triviwan.train`。不要直接把本项目的 `dataset.json` 传给原 `TriWorldDataset` 或 `triviwan.cache`。

## 1. 上传代码和转换结果

先在 LRZ 登录节点创建代码上传目录：

```bash
mkdir -p /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/triviwan_repro/triworld_training
```

在 Windows PowerShell 中，把项目代码上传到既有 DSS 项目目录内的独立目录。下列 `triworld_training` 是新目录名，可按需修改；上传转换好的数据目录后，确认 Linux 中该目录确实含 `dataset.json`、`windows.jsonl` 和任务子目录中的 `episodes/`（单任务清单的目录自身包含 `episodes/`）。训练数据不随 GitHub 代码仓库提交。

```powershell
scp -r "A:\Triworldbench训练数据集\triworld_train" "A:\Triworldbench训练数据集\scripts" "A:\Triworldbench训练数据集\configs" di97fer@login.ai.lrz.de:/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/triviwan_repro/triworld_training/
# 将实际转换输出目录上传到上述目录下，例如 training_data/。
scp -r "A:\Triworldbench训练数据集\datasets\train48" di97fer@login.ai.lrz.de:/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/triviwan_repro/triworld_training/training_data
```

第二条命令要求远端 `training_data` 尚不存在，此时 `scp` 会将 `train48` 内容复制到这个名称下。若远端已有数据，给新数据选择另一个目录名，避免 `scp` 产生 `training_data/train48/` 的额外层级。若 Windows 输出目录不同，以实际转换命令的 `--output-root` 为准；第二条命令中的 `datasets\train48` 是当前已生成的输出。上传后检查 `training_data/dataset.json` 路径，再填写训练配置。

## 2. 取得原模型代码

在 LRZ 登录节点执行。脚本仅解析部署文件内的字面量，检查 ZIP 摘要和路径后解包；不会运行部署文件的 Slurm 提交逻辑，也不会下载权重。只解包不需要 GPU、PyTorch 或模型环境。

```bash
export TRIVIEW_BASE=/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/triviwan_repro
export DATA_PROJECT="$TRIVIEW_BASE/triworld_training"
mkdir -p "$DATA_PROJECT"
cd "$DATA_PROJECT"
python3 scripts/prepare_lrz_cache.py --extract-only \
  --model-file "$HOME/TriViewWan_LRZ.py" \
  --model-extract-dir "$DATA_PROJECT/native_model"
export MODEL_ROOT="$DATA_PROJECT/native_model/TriViewWan"
```

`--extract-only` 支持 Python 3.10+；真实编码/训练使用原模型 `tomllib`，需要 Python 3.11+，已有计算容器的 `venv-py311-torch28/bin/python` 满足此要求。如果登录节点 Python 较旧，可在本机先执行同样的 `--extract-only`，再上传解出的 `native_model/`。也可以直接将 `MODEL_ROOT` 指向已部署的 `releases/single-版本摘要/TriViewWan`，并在编码命令使用 `--model-root`。解包目录已存在时，脚本逐文件检查同版本内容；不同版本或用户修改会要求另选目录。

## 3. 修改训练配置

编辑 `configs/lrz_train48.toml`。`paths.wan_source` 与 `paths.checkpoint` 已填写现有 LRZ 部署的源码和权重位置，仍应确认文件存在。将以下三个 `/CHANGE_ME` 值替换为 Linux 上的真实绝对路径：

```toml
[data]
manifest = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/triviwan_repro/triworld_training/training_data/dataset.json"
cache_dir = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/triviwan_repro/triworld_training/cache/train48_v1"
# 保留本节其余配置。

[train]
output_dir = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/triviwan_repro/triworld_training/runs/train48_v1"
# 保留本节其余配置。
```

上面的节选用于修改原文件相应字段，不应直接替代整份配置。先保持 `height=256,width=320,window_future=48,history.num_frames=9,action_dim=14`。当前原始数据缺乏可靠采集 FPS，契约为 `time_basis="frame_index",fps=null`；训练不会伪造物理秒数。播放 MP4 的 FPS 不属于采集频率。

动作仍保留原始数值；缓存编码时通过原模型 `fit_actions` 从每条完整训练轨迹拟合一次第一/第九十九百分位。重叠窗口与历史补帧均不重复参与统计，验证集不参与。训练器按原规则将动作映射到 `[-1,1]`、裁剪离群值、恒定维归零。`cache/action_normalizer.json` 是模型训练使用的准确 FP32 统计。

## 4. 在已分配 GPU 的原容器内编码和训练

缓存编码与训练应在计算节点、已有 BF16 GPU 和原模型容器中执行。下列变量使用既有容器和 Python；`SLURM_JOB_ID` 必须来自已获分配的作业。上传目录应位于 `TRIVIEW_BASE` 内，以保持容器同路径挂载。

```bash
export BWM_BASE=/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/bwm_repro
export LRZ_IMAGE="$BWM_BASE/containers/bwm-env-5813503.sqsh"
export LRZ_PYTHON="$TRIVIEW_BASE/venv-py311-torch28/bin/python"
export CONFIG="$DATA_PROJECT/configs/lrz_train48.toml"

srun --jobid="${SLURM_JOB_ID:?请先获取 GPU 分配}" --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  --container-image="$LRZ_IMAGE" \
  --container-mounts="$BWM_BASE:$BWM_BASE:ro,$TRIVIEW_BASE:$TRIVIEW_BASE" \
  --container-workdir="$DATA_PROJECT" --no-container-mount-home \
  env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV PYTHONNOUSERSITE=1 \
  "$LRZ_PYTHON" -u scripts/prepare_lrz_cache.py --config "$CONFIG" --model-root "$MODEL_ROOT"

srun --jobid="${SLURM_JOB_ID:?请先获取 GPU 分配}" --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  --container-image="$LRZ_IMAGE" \
  --container-mounts="$BWM_BASE:$BWM_BASE:ro,$TRIVIEW_BASE:$TRIVIEW_BASE" \
  --container-workdir="$MODEL_ROOT" --no-container-mount-home \
  env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV PYTHONNOUSERSITE=1 \
  "$LRZ_PYTHON" -u -m triviwan.train --config "$CONFIG" --stop-after 2
```

第二条先完成两次真实更新并保存停点 checkpoint；配置 `train.steps=1000` 仍保留完整学习率日程。确认 loss 和日志有限后，用相同配置续训：

```bash
# 在上面的训练 srun 命令末尾，把 --stop-after 2 换为以下参数。
--resume "$DATA_PROJECT/runs/train48_v1/step_0000002.pt"
```

正式多卡训练可在原模型目录中执行 `python -m torch.distributed.run --standalone --nproc_per_node=4 -m triviwan.train --config "$CONFIG"`；同时应申请四卡并在 `srun` 使用 `--gres=gpu:4`。新实验单卡与多卡不能混用精确恢复 checkpoint，原训练器会检查 world size。更改配置、动作统计或索引后应另建缓存和实验。

可用 Slurm 新建分配，例如在登录节点执行：

```bash
salloc --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:1 \
  --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-hgx-h100-94x4 --time=04:00:00
```

这只是资源申请命令，排队时长、所需 account 及允许的分区以账户当前配置为准。先前 LRZ 实测的是 9 历史 + 16 未来，不能据此保证 48 未来的显存或吞吐；本次需从 48 未来的首次编码和两步训练测量。显存不足时先降低空间尺寸（仍为 32 的倍数）并另建缓存，或减少梯度累积以降低等效批量；梯度累积通常不会明显降低单窗口峰值显存。

## 5. 缓存完整性与新数据

成功缓存包含 `index.jsonl`、`metadata.json`、`action_normalizer.json` 与逐窗口 `.pt`。`.pt` 内保留原训练器需要的 `clean,first,context,null_context,actions,weights,horizon,history_frames`，版本为 2。编码器为真实冻结 T5/VAE，不以零张量、首帧复制或占位 latent 代替未来监督。

脚本只在全部窗口成功后发布 `index.jsonl`。失败目录会保留 `BUILDING.json`/`FAILED.json` 和已写文件，原训练器不会把未发布目录当作可训练缓存。重新运行请给配置选择新的 `cache_dir`，避免混用部分结果。原模型训练器检查数据配置、manifest 摘要、原 Wan 源码摘要、历史规则和统计；适配器另记录窗口索引摘要及原模型关键代码摘要，方便核查。

新增 RoboTwin 任务转换后，使用新的完整数据集重新编码；统计从全部本次训练 episode 拟合。已有四轨迹旧 checkpoint 的统计不同，原训练器不允许直接 `--init-checkpoint` 热启动，需建立明确的统计迁移方案或从相同官方基座开始新实验。已有独立验证集继续按原模型推理/评测入口使用，不参与本项目转换统计。
