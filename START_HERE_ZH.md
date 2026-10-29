# 开源复现包使用入口

本包对应V53稿件，正式版本为`1.0.0`。它不包含原开发仓库的Git历史、旧稿、
审稿记录或文献库。

- 公开仓库：https://github.com/SII-Yuning-Zhou/finite-mode-semantic-coding
- 固定版本：https://github.com/SII-Yuning-Zhou/finite-mode-semantic-coding/releases/tag/v1.0.0

目前没有Zenodo DOI；引用和复现请使用上面的固定版本入口。

## 先运行这条轻量CPU复现路径

在解压目录打开终端，使用Python 3.11或更新版本：

```text
python -m pip install -r requirements.txt
python run_reproduction.py
```

建议先创建独立虚拟环境。命令验证理论见证、检查正式MNIST归档记录、冻结文件和端点账本并重绘数值图，
不训练网络、不下载MNIST、不启用GPU。结果写入`outputs/`，不覆盖随包数据。

## 现在包含的材料

- Bernoulli精确求解、严格四模式见证、非Bernoulli支持界见证及快速回归测试。
- 90个仿真点的正式均值/CI档案及完整生成参数；它不是逐符号原始流。
- MNIST全部100个调度级配对观测、20组汇总及正式协议报告，含L=1控制。
- 六张论文原图与独立、可移植的科学重绘入口；重绘保留数值，不宣称逐像素一致。
- 神经训练、条件rANS、聚合和端点profiling源码及正式配置。
- 四个原始codec checkpoint、冻结分类器权重、八份端点profile（16000次计时）及配置和80行账本。
- 英文README、CITATION、已确认的MIT/CC BY 4.0许可、环境与来源说明。

## 冻结权重与端点材料

服务器导出的26个文件现已补齐，原始checkpoint按`results/iclr_mnist/`下的原相对
路径保存。JSON仅去除私人绝对路径及内部标识，数值、种子、硬件和原始指纹保留；来源和修改
清单见`data/neural/frozen_artifacts_manifest.json`。

`python neural/verify_frozen_artifacts.py`验证这些冻结材料及原报告关联。
Table V从八份原始profile重聚合，同时保留原稿转录作为对照。重聚合不是重新测量时延。
完整神经评估需要读者准备MNIST和适配环境；详见`docs/NEURAL_REPRODUCTION.md`。

## 验证记录与投稿填写

查看`validation/validation_report.json`中的实际测试结果和SKIP项。
`docs/PUBLISHING_CHECKLIST_ZH.md`提供固定版本链接及代码/数据可用性声明。
投稿系统的“正文包含链接”应以稿件是否实际加入该链接为准；DOI栏保持空白，
除非后续已取得真实DOI。
