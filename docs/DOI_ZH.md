# 为 v1.0.0 获取 DOI

GitHub 提供可浏览的代码仓库和固定版本下载；Zenodo 可以为这个软件／数据发布包
提供可引用的 DOI。该 DOI 属于复现材料，不是论文发表后由期刊分配的论文 DOI。

## 本版本建议：手动归档同一份发布包

1. 在 https://zenodo.org 登录本人账号，选择 **New upload**。
2. 上传 GitHub Release 的完整 `finite-mode-semantic-coding-v1.0.0.zip`，
   可同时上传同名 `.sha256` 校验文件。
3. 资源类型选择 **Software**，版本填 **1.0.0**；标题、八位作者及单位沿用随包
   `CITATION.cff`。日期使用该版本实际发布日期，不填写虚构的期刊录用信息。
4. 本软件包尚未分配 DOI 时，在“Do you already have a DOI?”选择 **No**。
   可以在草稿中预留 DOI；点击 **Publish** 后才正式注册。不要把论文 DOI 填在这里。
5. 许可保留两个适用范围：原代码及使用文档为 **MIT**；作者生成的数据、训练权重和
   图稿为 **CC BY 4.0**。第三方软件、MNIST和字体继续遵守其原许可。
6. 在描述／相关标识符中关联实际 GitHub 仓库及 `v1.0.0` Release；检查文件、作者、
   版本及许可后由作者点击 **Publish**。
7. 取得该版本 DOI 后核对落地页和文件，把真实 DOI 加入 README、CITATION 及论文的
   代码／数据可用性声明。不要用 GitHub 网址代替 DOI。

仓库：https://github.com/SII-Yuning-Zhou/finite-mode-semantic-coding

固定版本：https://github.com/SII-Yuning-Zhou/finite-mode-semantic-coding/releases/tag/v1.0.0

## 可复制的 Zenodo 描述

```text
Code and reproducibility materials accompanying "Finite-Mode Semantic Coding
for Edge IoT under Target Distribution and Shared-Randomness Constraints".
Version 1.0.0 includes theoretical solvers and certificates, numerical and
figure source data, paired MNIST conditional-coding reports, five original
frozen checkpoints, and eight endpoint profiles containing 16,000 recorded
timing calls. Original software and usage documentation are licensed under MIT.
Author-generated numerical data, trained weights, and graphics are licensed
under CC BY 4.0. Third-party resources retain their original terms. MNIST images
are obtained separately through torchvision and are not redistributed here.
```

## 后续版本自动归档

也可以在 Zenodo 授权连接 GitHub，并启用本仓库；启用之后的新 GitHub Release 可自动
归档。这个操作涉及第三方应用授权，应由账号持有人确认权限。已发布的 v1.0.0 可以按
上面的手动流程归档，无需删除 Release 或为了获取 DOI 虚构一个新软件版本。

Zenodo 的具体版本 DOI 与所有版本的 Concept DOI 不同；论文的精确复现引用宜指向
实际使用的 v1.0.0 版本 DOI。后续材料更新在同一 Zenodo 记录下创建 **New version**。

官方说明：

- 创建记录：https://help.zenodo.org/docs/deposit/create-new-upload/
- 预留 DOI：https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/
- 混合许可：https://help.zenodo.org/docs/deposit/describe-records/licenses/
- 手动软件归档：https://help.zenodo.org/docs/github/archive-software/manual-upload/
- GitHub 联动：https://help.zenodo.org/docs/github/enable-repository/

本发布操作没有代替作者创建 Zenodo 账号、授予第三方访问或发布 Zenodo 记录；没有
在文件中虚构 DOI。
