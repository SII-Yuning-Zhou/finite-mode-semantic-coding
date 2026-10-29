# 公开版本与投稿填写

## 正式版本与入口

- 已获作者确认：自编代码MIT，作者生成数据CC BY 4.0；第三方材料维持原许可。
- 版本：`1.0.0`。公开仓库：https://github.com/SII-Yuning-Zhou/finite-mode-semantic-coding 。
- 固定版本：https://github.com/SII-Yuning-Zhou/finite-mode-semantic-coding/releases/tag/v1.0.0 。
- 本包使用独立Git历史；原开发仓库及其可见性保持不变。
- 目前没有Zenodo DOI；GitHub网址不是DOI，不应填入DOI栏。
- 四个codec、冻结分类器权重及八个原始edge profile均包含在本版本；来源及路径转换
  记录见`data/neural/frozen_artifacts_manifest.json`，实际检查结果见`validation/`。

## 后续维护与归档

1. 检查随包的来源与验证记录。科学参数、种子、硬件型号和测量值保留；JSON私人
   路径转换有清单。原始checkpoint bytes未改；权重指纹可与正式报告核对。
2. 在适配的PyTorch/CompressAI环境中按需运行冻结权重完整评估和新设备profiling。
   已通过的轻量复现和CPU smoke检查见`validation/`；它们不等于重跑全部训练或测量。
   不将SKIP改写成PASS，不将新训练当作原冻结checkpoint。
3. 核查`LICENSE`、`LICENSE-DATA.md`和第三方说明覆盖范围。公开的是本文复现材料，
   无需一并公开参考文献PDF、草稿或审稿记录。
4. 后续修改在公开仓库中建立新版本；保留`v1.0.0`固定tag及其归档材料，
   不将原开发仓的整个工作目录、历史、草稿或审稿记录混入公开仓。
5. 检查读者在未登录作者账号时也可下载固定release中的数据和所需模型。
6. 如采用Zenodo，归档该固定release取得真实DOI；检查DOI落地页、许可、作者、文件和版本。
7. 把实际URL/DOI加入README、CITATION和论文的代码/数据可用性声明，再填写投稿系统。

GitHub官方归档说明：
https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content

## 投稿系统字段

- “正文包含链接”：真实入口已加入稿件后再勾。
- “已上传数据”：数据已实际发布、审稿人可以访问后再勾。只有代码不等于数据齐全。
- DOI栏填实际DOI，不填GitHub网址，也不编造编号。
- 数据仓库名称应与该记录的实际存放位置一致。

## GitHub版本的代码/数据可用性声明

```text
The code, figure source data, original trained checkpoints, paired coding-audit
records, and endpoint profiling records supporting this study are available
in the public reproducibility package (version 1.0.0) at
https://github.com/SII-Yuning-Zhou/finite-mode-semantic-coding/releases/tag/v1.0.0.
MNIST is obtained from its original public source through torchvision.
```

确认固定版本链接可公开访问后，可将以上声明加入稿件。若后续取得Zenodo DOI，
再补充真实DOI和归档入口；不要在此前写成“已通过Zenodo归档”。
