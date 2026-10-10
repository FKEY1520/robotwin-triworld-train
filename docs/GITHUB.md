# 将完整项目上传 GitHub

仓库提交转换器、配置、测试、示例和文档。大体积源数据、生成的 `datasets/`、导出 NPZ、LRZ 缓存、模型权重和 `.venv/` 留在本地或数据存储中，已由 `.gitignore` 排除。

本机项目已初始化 Git，开发分支为 `codex/train-dataset-48`。准备首次推送时可按下面的命令将它重命名为 `main`。远端仓库与 GitHub 可见性由你选择。

## 本地检查

在 PowerShell 中进入项目目录：

```powershell
Set-Location -LiteralPath 'A:\Triworldbench训练数据集'
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe .\convert_dataset.py --help
```

新副本尚未初始化 Git 时：

```powershell
git init -b main
```

若项目已经包含 `.git`，跳过初始化。然后检查待提交内容：

```powershell
git status --short
git add .
git diff --cached --stat
git diff --cached --name-only
```

文件列表中应出现 Python、JSON、TOML、Markdown、测试和工作流，不应出现 `.h5`、`.npz`、`.venv` 或数据图片。可检查忽略规则：

```powershell
git check-ignore -v .venv/ datasets/train48/
```

提交：

```powershell
git commit -m "Add Robotwin future-48-frame training dataset pipeline"
```

## 在 GitHub 创建仓库并推送

打开 GitHub，新建仓库，例如 `robotwin-triworld-train`。选择可见性；若要直接推送本地已有项目，新建时不要勾选自动生成 README 或 `.gitignore`。将下列 URL 替换为你实际的账户和仓库：

```powershell
git remote add origin https://github.com/YOUR_ACCOUNT/robotwin-triworld-train.git
git branch -M main
git push -u origin main
```

若已经存在 `origin`，先用 `git remote -v` 查看地址，再根据需要使用 `git remote set-url origin <实际地址>`。不要把这里的占位账户直接运行成正式地址。

安装并登录 GitHub CLI 后，也可用它代替网页创建仓库（以下例子为私有仓库）：

```powershell
gh repo create robotwin-triworld-train --private --source . --remote origin --push
```

该命令会实际创建远端仓库并推送。根据自己的仓库名称和可见性选择后再运行。本项目不会替你猜测 GitHub 账户或创建远端仓库。

## 开源许可

项目没有自动附加开源许可证。请在发布前按代码所有权、使用计划和相关来源要求选择许可证，添加到 `LICENSE`。公开仓库的可见性与许可授权是两件不同的事。

## 数据单独交付

本格式可以单独搬运：

```text
datasets/train48/                       # 所有任务及汇总索引
datasets/train48/adjust_bottle_clean50/  # 一个任务的独立数据集
```

完整复制选中的目录及其子目录到 LRZ 或其他存储后，运行：

```bash
.venv/bin/python convert_dataset.py verify --dataset-root /path/to/train48 --deep
```

保存对应的 `normalizer.json` 和 `report.json`。跨机器读取依赖相对路径及 HDF5 内部图片，不要求源 `A:` 盘存在。若希望未来公开数据，可以选择独立的数据发布平台并在 README 中增加下载位置；不要将大数据直接加入本代码仓库。

## 后续修改与发布

```powershell
git status --short
git add .
git commit -m "Describe your change"
git push
```

GitHub Actions 会自动运行 Python 3.10/3.12 测试。正式数据版本建议同时保存转换代码 commit 与配置文件；每次数据定义变更使用新的输出目录，避免训练结果无法追溯。
