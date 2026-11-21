# 四模态内容真实性检测

文字、图像、音频、人脸视频检测，支持基础方法与论文方法对照、校准置信度、证据、误报漏报、普通扰动与白盒对抗自评。

| 模态 | 基础方法 | 论文方法 |
|---|---|---|
| 文本 | TF-IDF＋逻辑回归 | Fast-DetectGPT解析曲率，DistilGPT2轻量适配 |
| 图像 | FFT、颜色、梯度统计＋逻辑回归 | UnivFD，CLIP ViT-L/14与官方线性头 |
| 音频 | LFCC统计＋逻辑回归 | AASIST-L |
| 视频 | 人脸帧特征统计＋逻辑回归 | DeepfakeBench FF++ Xception |

## 安装与资产

建议Python3.11以上，先安装适合CPU或CUDA的PyTorch与torchvision，再安装requirements-autodl.txt（Linux）或requirements.txt。

仓库只保存源码、必要配置、测试和第三方许可证。运行前需恢复服务器已有数据、清单、权重、基础分类器、校准及运行时审计配置。原始数据、课程课件、模型文件、实验结果和私人交接材料不发布到Git。

## 运行

```bash
export LAB_DEVICE=cuda  # 无GPU时设置cpu
export LAB_THREADS=4
python app.py
```

默认访问 http://127.0.0.1:7860 。基础方法的NumPy/scikit-learn运算仍在CPU。

```bash
python scripts/run_experiments.py --method all
python scripts/select_methods.py
python scripts/run_robustness.py --method paper
python scripts/run_attacks.py --modality image --method paper --profile full
python -m unittest discover -s tests -v
```

上述命令依赖完整运行资产；只克隆源码不代表已经完成部署。已有划分不要重新随机生成；更换方法或预处理时隔离旧分数和特征缓存。下载课程数据需自行设置LAB_COURSE_BASE_URL与LAB_COURSE_SHARE_ID，仓库不提供私人分享入口。

## AutoDL同步

本次是本地代码基线，服务器后来产生的更新尚待合并。请阅读[GITHUB_SYNC.md](GITHUB_SYNC.md)，在独立克隆和更新分支中比较、提交服务器修改，保留其数据、模型与结果。

第三方代码与许可证位于vendor。参考Fast-DetectGPT、UniversalFakeDetect、AASIST、DeepfakeBench的官方仓库。文本属于轻量适配；图像预处理与官方仅中心裁剪不同；视频采用Haar人脸裁剪。来源真实性检测不等于事实核查，梯度图也不是真实篡改位置。
