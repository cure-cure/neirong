# 实验六：四模态内容真实性检测与自评

胡逸辰 · 2024302181004

## 启动本地界面

在项目目录打开 PowerShell：

```powershell
.\.venv\Scripts\python.exe app.py
```

浏览器访问 http://127.0.0.1:7860 。也可运行 `启动系统.ps1`。

四种模态默认使用验证集选定的方法。基础、论文和等权融合仍可手动比较。论文方法较差时不会强制用于最终检测。当前选择与理由在 `configs/selected_methods.json`；使用验证集的平衡准确率选模型，同分优先基础方法。文本另外使用独立外部验证划分，按领域等权选择；最初外部200条文本是诊断集，新采集的另200条test文本才是后续最终保留测试。

首次加载论文模型较慢；切换模态会释放上一个深度模型。所有推理使用本地权重，不需要付费API。视频检测面向人脸伪造，当前均匀抽8帧、取每帧最大人脸；无可用人脸会明确报错。英文短文本、中文与非人脸视频不在已验证的主要范围。

## 目录

| 目录 | 内容 |
|---|---|
| src | 解码、基础特征、论文模型、概率校准、推理、扰动、白盒攻击 |
| scripts | 数据采集、训练评估、默认方法选择、普通扰动、自评、报告生成 |
| configs | 默认方法与迁移配置 |
| data/raw/course | 用户提供分享的原始348样本及说明文件 |
| data/manifest.json | 标签、来源、组别、训练/验证/测试划分、SHA256 |
| data/raw/external | 自主下载的公开补充测试数据（如已完成下载） |
| models | 官方权重、本地训练基础模型、校准参数、来源记录 |
| vendor | 官方模型代码及许可证；Xception去掉注册器的适配版与原版并存 |
| results | 逐样本结果、混淆矩阵、普通扰动、对抗前后样本、UI现场记录 |
| reports | 按真实结果生成的实验报告 |
| references | 原始课程PPT |

## 在另一台机器配置

建议Python 3.11或3.13，并使用独立环境。本机实际使用3.13.5，复用已有CPU PyTorch；不修改系统已有实验环境。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果新建虚拟环境的ensurepip受限，可先 `python -m venv --without-pip --system-site-packages .venv`，再通过已有pip执行 `python -m pip --python .venv/Scripts/python.exe install -r requirements.txt`。本机采用此复用方式。

## 重跑实验

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe scripts/run_experiments.py --method all
.\.venv\Scripts\python.exe scripts/select_methods.py
.\.venv\Scripts\python.exe scripts/run_robustness.py --method baseline
.\.venv\Scripts\python.exe scripts/run_robustness.py --modality audio --method paper
.\.venv\Scripts\python.exe scripts/run_attacks.py --modality audio --method baseline --profile full
.\.venv\Scripts\python.exe scripts/run_attacks.py --modality audio --method paper --profile local
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/build_report.py
```

执行全量论文评估可能耗时较长。脚本逐样本保存检查点，已有成功记录会跳过；修改方法、预处理或权重后应先把旧的对应结果移动到归档目录，再重跑，避免混合不同配置。普通扰动支持 `--limit N`，限制只是快速验证，不能冒充完整评估。对抗脚本支持 `--limit N` 和 `--profile local/full`。

补充数据可通过 `scripts/download_external.py` 下载，之后用 `scripts/evaluate_external.py --modality text --method baseline` 或 `--modality audio --method paper` 评估。补充集不参与训练、校准、模型选择。音频是同语料扩展，不能称为跨语料泛化；文本来自不同来源的GPT-2官方测试集。

## 视觉外部测试与全量攻击更新

当前项目另含固定CIFAKE外部图像子集：官方test划分中100张CIFAR-10真实图和100张Stable Diffusion 1.4伪造图。来源、固定镜像提交、选样规则与逐文件哈希见 `data/external_visual_source.json` 和 `data/external_visual_manifest.json`。重新评价命令：

```bash
LAB_DEVICE=cuda python scripts/evaluate_external.py --modality image --method selected --manifest data/external_visual_manifest.json --split external_test --name cifake_external
```

图像和视频论文模型的full攻击已经分别完成480和456条记录。视频请求20例、有效19例，唯一无人脸失败保存在 `results/attacks_video_paper_full_failures.json`。逐样本GPU证据独立保存在 `results/gpu_full_attack_examples/`，未覆盖原CPU证据。报告现包含可靠性图、95%区间、覆盖率和失败计错准确率。

注意：CIFAKE默认方法准确率仅48%，FPR为99%；该结果应作为32×32分辨率和来源域偏移下的失败证据，不得改阈值或删样本美化。

## AutoDL/Linux迁移

迁移项目源码、configs、models、vendor、data与所需results；**不要复制Windows的.venv**，也不需要下载临时分块目录。Linux环境重新安装依赖，按实际CUDA环境安装官方PyTorch CUDA版本。代码不依赖D盘路径。

```bash
python -m venv .venv
source .venv/bin/activate
# 先按GPU驱动兼容情况安装官方CUDA PyTorch，再装业务依赖。
pip install -r requirements.txt
export LAB_DEVICE=cuda
export LAB_THREADS=4
python scripts/run_experiments.py --method all
python scripts/run_attacks.py --modality image --method paper --profile full
python scripts/run_attacks.py --modality audio --method baseline --profile full
python scripts/build_report.py
```

GPU服务器默认仍只监听127.0.0.1。通过SSH端口转发访问UI：`ssh -L 7860:127.0.0.1:7860 <你的服务器连接参数>`，再在本机访问本地7860端口。无需把服务公开到互联网。

## 实验口径

- 伪造为正类，阈值0.5。置信度是验证集校准估计，小样本下可能不可靠。
- UI的“存疑”与二分类测试口径分开，未检出人脸不归为真实。
- 测试成绩只使用留出集；原始论文预训练可能与公开数据有重合，未声称完全排除。
- 论文模型与基础方法的比较同时保留原始和校准结果。校准正斜率，不翻转标签方向。
- FGSM/PGD作用于检测器输入；音频基础方法攻击完整波形，论文音频攻击首个4.04秒片段；视频攻击首个人脸裁剪。注意区分文件级与模型输入级结论。
- 平滑防御重新生成穿过防御的攻击。ASR分母只包含原本预测正确的样本；为0时记录null。
- 不把压缩频谱、声谱图或显著性线索当成已证明的篡改区域。
- 原始数据和攻击证据可能包含公开人物及新闻材料，只用于课程研究与检验。

## 主要论文

- Fast-DetectGPT，ICLR2024：https://github.com/baoguangsheng/fast-detect-gpt 。本地使用DistilGPT2轻量适配配置，非原论文大型模型实验。
- UnivFD，CVPR2023：https://github.com/WisconsinAIVision/UniversalFakeDetect 。官方CLIP ViT-L/14和线性头。
- AASIST，ICASSP2022：https://github.com/clovaai/aasist 。官方AASIST-L权重。
- FaceForensics++ Xception／DeepfakeBench：https://github.com/SCLBD/DeepfakeBench 。使用已训练伪造检测权重，不是ImageNet初始化权重。
- OpenTuringBench，EMNLP2025：https://aclanthology.org/2025.emnlp-main.1354/ 。作为课程文本来源与评估设计参考，未冒称实现OTBDetector。

报告以当前已生成结果为准。已完成图像/视频全量GPU攻击与CIFAKE外部图像评估；外部视频评估和对抗训练仍未完成，不得写成已完成。
