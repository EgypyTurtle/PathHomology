# 更新原仓库 EgypyTurtle/PathHomology

目标始终是 **https://github.com/EgypyTurtle/PathHomology**，不需要新建仓库，不需要重新添加 origin。

本次 gen6/Morse 更新基于 main 的 `2a226be`，保留原 Git 历史、families、文献校验、原始核心和历史结果。主 README 在旧版基础上扩充，而非另换一份说明。以下步骤适用于以后继续更新此仓库；本次提交与推送已由项目所有者另行明确授权。

## 直接使用这个本地克隆

在本目录打开终端，先检查：

```bash
git remote -v
git status --short
git diff --stat
python -m pip install -r requirements.txt
python -B verify_release.py
python -B verify_repository.py
```

确认 origin 为上述仓库，审阅新增文件及 README 后，由你提交和推送：

```bash
git add .
git commit -m "Add gen6 algebraic Morse backend and reproducible validation"
git pull --rebase origin main
git push origin main
```

若远端在整理期间有新提交，pull/rebase 可能要求你解决冲突；遇到冲突先处理，不要强制推送。此次没有替你设置 Git 作者身份。

`.gitattributes` 保留源码原始字节，避免换行转换破坏历史性能记录的 SHA256；不要因使用 Windows 自动删掉它。

## 继续用你自己已有的本地仓库

不必改用这个独立克隆。也可以把更新 ZIP 的内容合入你原来的仓库根目录，保留你自己的 `.git`，正常审阅和提交。若你自己的本地副本已有未提交修改，先备份或提交它们；README/core 等同名文件需要审阅，不能盲目覆盖。

本更新 ZIP 包含原仓库的 families、文献脚本、历史结果和新的核心/文档；不包含 `.git`、缓存、`.local-results` 或虚拟环境。它是给**原仓库**的更新文件集合，不是一份要重新初始化的空仓库。

## 使用 GitHub 网页上传

进入原仓库，解压 ZIP 后通过 Add file → Upload files 上传目录内的文件。目录本身不是一个新的仓库层级，不要把整个 `PathHomology_update_ready/` 作为嵌套子目录上传，也不要只上传 ZIP 文件。

请包含 `.github/workflows/tests.yml`、`.gitignore`、`.gitattributes`。不要删除已有的 `families/`、`results/`、文献校验脚本或原始核心。

## 许可

原仓库尚无明确 LICENSE，本次未代选。如果你打算指定开源许可证，请由你确认后补充 LICENSE。这不影响当前本地整理与验证已经完成。
