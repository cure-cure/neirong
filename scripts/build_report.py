import sys,json,datetime,html
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

REFERENCES=[
('课程要求','references/实验6_多模态内容安全检测系统设计与真实性评估.pptx'),
('数据集说明','references/内容安全实验_数据集介绍 (1).pptx'),
('OpenTuringBench, EMNLP2025','https://aclanthology.org/2025.emnlp-main.1354/'),
('Fast-DetectGPT, ICLR2024','https://github.com/baoguangsheng/fast-detect-gpt'),
('UnivFD, CVPR2023','https://github.com/WisconsinAIVision/UniversalFakeDetect'),
('AASIST, ICASSP2022','https://github.com/clovaai/aasist'),
('FaceForensics++, ICCV2019','https://github.com/ondyari/FaceForensics'),
('DeepfakeBench (Xception weights and architecture)','https://github.com/SCLBD/DeepfakeBench')]

def main():
    from build_evidence import main as build_evidence
    build_evidence()
    summaries=[read(p) for p in sorted((ROOT/'results').glob('*_metrics.json'))]
    lines=['# 实验六：多模态内容安全检测系统设计与真实性评估','',
    '姓名：项目作者　学号：STUDENT_ID','',
    '本报告由实际结果文件生成。报告中的二分类以伪造为正类；不把 AI 来源检测视为事实核查。实验范围与尚未完成项在文末列明。','',
    '## 1. 实验目标与系统结构','',
    '支持文字、图像、音频、视频四种输入。输入解码后分别提取基础特征和执行论文模型推理，用验证集进行正斜率逻辑概率校准，再比较基础方法、论文方法与等权平均方法。输出判定、置信度、证据、真实性评分及风险等级。','',
    'UI 展示预处理、特征线索、逐段／逐帧模型输出、运行参数与结果记录。跨模态采用等权融合，缺失模态重新归一化，同时保留最大分项风险和投票比例。课程数据不构成天然配对的四模态材料，因此不能声称已经证明跨模态融合的准确率。','',
    '## 2. 数据与防泄漏','',
    '课程分享包含 348 个样本：文本 100、图像 100、音频 50、视频 98。原始文件均保留，data/course_sources.json 保存来源、字节数与 SHA256，data/manifest.json 保存标签、组别和划分。划分固定随机种子 202430，按组约 60%/20%/20% 划为训练、验证、测试。','',
    '文本与视频同编号真假样本放在同组，扰动版本继承原样本划分。音频缺少说话人身份，无法保证说话人独立；视频身份独立性也未知。训练与验证结果不能混入最终测试。所有论文检测器的外部预训练数据与当前教学子集是否重合仍需审查。','',
    '```json',json.dumps(read('results/data_audit.json')['splits'],ensure_ascii=False,indent=2),'```','',
    '## 3. 基础方法与论文方法','',
    '| 模态 | 基础方法 | 论文方法与实际适配 |','|---|---|---|',
    '| 文本 | TF-IDF 单词1–2 gram + 逻辑回归，C=1 | Fast-DetectGPT 解析条件概率曲率；用 DistilGPT2 同时作为参考与评分模型，上限256 token。这是轻量适配，不是原论文的大模型配置，也未实现 OTBDetector |',
    '| 图像 | 16维径向傅里叶统计、RGB均值方差、梯度统计 + 标准化逻辑回归，C=0.1 | UnivFD 官方 CLIP ViT-L/14 与官方线性头，短边224后中心裁剪，使用 CLIP 归一化 |',
    '| 音频 | 16kHz、400点窗、160点移、24线性滤波器、20维LFCC均值/方差/差分绝对均值 + 标准化逻辑回归 | 官方 AASIST-L；64600点分段，短片段重复填充，官方类别0为spoof；全音频取片段概率均值 |',
    '| 视频 | Haar最大人脸裁剪、8个均匀采样帧、图像基础特征的均值方差 + 标准化逻辑回归 | DeepfakeBench 发布的 FF++ Xception 检测权重，RGB 256×256，归一化[-1,1]，帧概率均值 |','',
    '全部深度权重严格加载，未使用随机初始化网络冒充检测器。Xception 只去掉框架注册器，架构未修改。第三方代码与许可证在 vendor 目录，下载记录见 models/sources.json。','',
    '文本曲率公式：将真实下一个 token 的对数概率与参考分布下的期望相减后求和，再除以方差之和的平方根。分数越高越倾向生成文本；截断部分在结果中标明。','',
    '校准只使用验证集，形式为 sigmoid(a×score+b)，约束 a>0，避免通过翻转论文模型方向掩盖跨域失败。基础与论文分数各自校准后等权平均，不根据测试集挑选融合权重。','',
    'UI 的存疑规则为置信度低于0.65，或两方法伪造概率相差超过0.5。测试指标统一使用0.5阈值的二分类结果，存疑不是第五种真值类别。小验证集的概率可靠性有限。','',
    '## 4. 留出测试结果','',
    '| 模态 | 方法 | 有效测试数 | Accuracy | Precision | Recall | F1 | 误报率 | 漏报率 | ROC-AUC | Brier |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    records=[]
    for s in summaries:
        t=s['test'];row=dict(modality=s['modality'],method=s['method'],**t);row.pop('confusion_matrix',None);records.append(row)
        lines.append('| '+MODALITIES[s['modality']]+' | '+s['method']+' | '+str(t['n'])+' | '+' | '.join(f'{t[k]:.4f}' if t[k] is not None else 'N/A' for k in ['accuracy','precision','recall','f1','fpr','fnr','roc_auc','brier'])+' |')
        fig,ax=plt.subplots(figsize=(4,3));cm=np.array(t['confusion_matrix']);ax.imshow(cm,cmap='Blues');ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Real','Fake'],yticklabels=['Real','Fake'],xlabel='Predicted',ylabel='Ground truth',title=s['modality']+' / '+s['method'])
        for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center')
        fig.tight_layout();p=ROOT/f'results/figures/{s["modality"]}_{s["method"]}_confusion.png';p.parent.mkdir(exist_ok=True);fig.savefig(p,dpi=130);plt.close(fig)
    if records:pd.DataFrame(records).to_csv(ROOT/'results/metrics.csv',index=False,encoding='utf-8-sig')
    lines+=['','成功处理样本上的指标与处理覆盖率分开报告。未检出人脸等失败不自动归为真实；有效准确率还应将处理失败作为错误计入总分母。详细覆盖率、原始分数指标和按组bootstrap准确率区间见各 metrics.json。样本很少，不能由某一次100%结果推出普适检测能力。','',
    '## 5. 误报漏报分析','']
    for s in summaries:
        score_path=ROOT/f'results/{s["modality"]}_{s["method"]}_scores.json'
        if not score_path.exists():continue
        errors=[r for r in read(score_path) if r['split']=='test' and ('error' in r or int(r['fake_probability']>=.5)!=r['label'])]
        lines.append(f'### {MODALITIES[s["modality"]]} / {s["method"]}')
        lines.append('')
        for r in errors:
            lines.append('- '+r['id']+'：'+('处理失败：'+r['error'] if 'error' in r else ('误报' if r['label']==0 else '漏报')+f'，伪造概率 {r["fake_probability"]:.4f}'))
        if not errors:lines.append('本次留出集未观察到错误；仍受测试规模与分布限制。')
        lines.append('')
    lines+=['AASIST 的原始跨域表现须与校准后结果同时阅读。若模型分数不能区分两类，校准只能调整概率，不能创造辨别能力。图像真假原文件分别主要为JPEG/PNG，基础频域方法可能学习压缩线索，需结合JPEG重压缩结果分析。文本的词汇与风格差异也可能成为数据集捷径，必须用新领域和新生成器继续验证。','',
    '## 6. 普通扰动鲁棒性','',
    '文本测试空格、标点、错字与截断；图像测试JPEG90/50、缩放、模糊、噪声与亮度；音频测试30/20/10dB噪声、音量、8kHz往返重采样、截断与变速；视频当前测试提取后人脸裁剪的压缩、缩放、模糊和亮度，不能替代完整视频重编码测试。截断可能移除伪造线索，解释时需区分输入变化与检测器脆弱性。','']
    for p in sorted((ROOT/'results').glob('robustness_*_summary.json')):
        obj=read(p);lines+=['### '+p.stem,'','| 条件 | 强度 | 样本数 | Accuracy | FPR | FNR |','|---|---:|---:|---:|---:|---:|']
        for r in obj['results']:lines.append(f'| {r["condition"]} | {r["level"]} | {r["n"]} | {r["accuracy"]:.4f} | {r["fpr"]:.4f} | {r["fnr"]:.4f} |')
        lines.append('')
    lines+=['## 7. 对抗自评与防御','',
    '攻击使用真实标签最大化二分类损失。FGSM单步、无随机初始化；PGD随机初始化，每步投影到L∞球并裁剪到合法值域。局部配置测试两个扰动强度与FGSM/PGD5，完整配置增加四个强度及PGD20三次重启。图像预算以[0,1]像素计，音频以[-1,1]波形幅值计。参数与实际L∞、L2、SNR均逐样本保存。','',
    '攻击成功率ASR = 原本正确且攻击后错误数 / 原本正确受攻击数。分母为0时记录null，不能填0；同时报告攻击后准确率，区分真实→伪造和伪造→真实。','',
    '平滑防御为3点／3×3均值滤波；防御实验重新穿过防御生成攻击，不能把旧攻击迁移结果当作白盒防御结果。防御保留原始干净验证集校准参数，未做防御特定校准。音频基础方法攻击完整波形，论文音频攻击第一个4.04秒模型片段，视频作用于第一个检测到的人脸裁剪；必须按各自威胁模型解释。音频基础方法另通过PCM16保存、重新读取检查攻击是否保留，量化误差另行计入，不能继续声称量化后严格满足原浮点预算。','']
    for p in sorted((ROOT/'results').glob('attacks_*_summary.json')):
        obj=read(p);lines+=['### '+p.stem,'','| 攻击 | ε | 防御 | 测试数 | 原本正确数 | ASR | 攻击后准确率 |','|---|---:|---|---:|---:|---:|---:|']
        for r in obj['results']:lines.append(f'| {r["attack"]} | {r["epsilon"]:.6f} | {r["defense"]} | {r["n"]} | {r["eligible"]} | {r["asr"] if r["asr"] is not None else "N/A"} | {r["accuracy"]} |')
        lines.append('')
    lines+=['## 8. 可复现性与运行','',
    '本地使用CPU，默认4个PyTorch线程，模型按需加载。UI现场推理记录在results/ui_runs，批量结果按模态独立保存。LAB_DEVICE=cpu/cuda/auto控制设备，所有路径相对项目根目录。迁移Linux时重新建立环境，不能复制Windows虚拟环境。','',
    '运行顺序、安装和迁移命令见README.md。单元验证覆盖来源哈希、分组无交叉、扰动预算与攻击方向、校准不翻转模型、官方权重严格加载。','',
    '## 9. 当前局限与后续工作','',
    '- 视觉模态结论基于教学子集；扩展文本和音频不能代替图像／视频的独立外部评估。','- 尚未收集用户本人拍摄／录制的真人材料，不能把公开下载数据写成个人实拍。','- 文本主要为英语，中文、新型生成器与人机混写需另建测试集。','- 视频采用最大人脸与均匀抽帧，无身份跟踪，不保证短时局部篡改均被采样。','- 音频基础特征受声学条件影响，AASIST存在跨数据集失效风险。','- 模型输入层攻击不等于编码、解码、人脸检测全流程攻击。','- 未完成的完整GPU扫描、跨域测试或对抗训练不能作为已完成实验列入成绩证据。','- 风险分数与显著性线索均不能证明某人造假或内容事实错误。','',
    ]
    lines+=['','## 10. 默认方法选择与论文失效分析','']
    selection_path=ROOT/'configs/selected_methods.json'
    if selection_path.exists():
        selection=read(selection_path)
        lines+=['| 模态 | 最终默认方法 | 课程验证集基础平衡准确率 | 课程验证集论文平衡准确率 |','|---|---|---:|---:|']
        for modality,obj in selection.items():
            values=obj['validation'];b=values.get('baseline',{}).get('balanced_accuracy');p=values.get('paper',{}).get('balanced_accuracy')
            lines.append(f'| {MODALITIES[modality]} | {obj["selected"]} | {b} | {p} |')
        lines+=['','文本额外使用独立外部验证数据，按领域等权平均平衡准确率选择默认方法；其余模态使用课程验证集。同分优先基础方法。融合仅为消融对照，不强制作为默认。选择理由与全部候选指标保留在configs/selected_methods.json。','']
    lines+=['### 为什么某些论文方法会较差','',
    '音频：官方AASIST-L主要在ASVspoof2019 LA训练，当前SecureAudio的说话人、合成算法、声道与录制条件可能不同。已复核官方类别0=spoof、类别1=bonafide、16kHz、64600点重复填充，严格加载权重，未发现由标签反转或随机初始化造成的失败。当前实际结果支持“跨域辨别力不足”：原始ROC-AUC、真实与伪造分数重叠、校准斜率和误报率均应结合阅读。具体由哪种声学差异主导尚未通过控制实验确定，不能把可能原因写成已证明结论。','',
    '文本：TF-IDF基础方法能捕捉课程AI文本中的词汇与写作习惯，课程小样本得分很高，但外部GPT-2文本表现下降。条件概率曲率方法依赖语言模型的token概率，减少了对固定词汇的直接依赖，跨来源结果更好。DistilGPT2仍是本项目的轻量适配，不能以此推断它对中文、现代推理模型或所有改写都有效。','',
    '图像与视频：以实际验证结果决定默认方法，不预设深度论文方法必胜。UnivFD使用官方权重，但本地采用标准CLIP短边缩放再中心裁剪；官方仓库validate.py使用仅中心裁剪，因此本项目不是逐参数复现。该适配在本报告中明确列出，不能直接对照论文表格宣称复现其精度。视频使用Haar最大人脸代替原框架完整对齐流程，同样存在预处理域差异。','',
    '防御：均值平滑对某些高频扰动可能有作用，但本项目对攻击穿过防御重新优化后观察到防御失效。这说明不能只展示一张视觉变化很小的攻击图片，也不能把基础方法正常检测较好等同于对抗鲁棒。','',
    '### 补充数据与冻结评估','',
    '| 实验 | 方法 | 样本数 | Accuracy | FPR | FNR | ROC-AUC |','|---|---|---:|---:|---:|---:|---:|']
    for pattern in ['external_*_summary.json','final_external_*_summary.json']:
        for p in sorted((ROOT/'results').glob(pattern)):
            obj=read(p)
            if 'metrics' not in obj:continue
            r=obj['metrics'];lines.append(f'| {p.stem} | {obj["method"]} | {r["n"]} | {r["accuracy"]:.4f} | {r["fpr"]:.4f} | {r["fnr"]:.4f} | {r["roc_auc"]:.4f} |')
    lines+=['','最初200条外部文本为诊断集。观察到来源偏移后，另从官方valid划分采集200条验证数据，并从test划分后续未使用索引采集200条新的最终测试数据。此为探索性协议扩展，过程完整记录在results/text_selection_protocol.json。模型训练和概率校准仍仅使用课程训练／验证划分；新最终测试不参与方法选择。补充音频排除了与课程样本的精确重复，但仍是同语料扩展，不代表未知说话人或未知合成器泛化已获证明。','',
    '### 证据图与交付清单','',
    '逐样本分数、异常、误报漏报、校准参数及攻击前后数组均保留。UI通过实际HTTP上传与队列接口测试四模态、联合评估、图表及结果下载；视觉检查工具初始化故障，因此接口通过不能替代页面布局截图验收。报告提供操作说明作为课件允许的演示材料。','']
    for s in summaries:
        lines+=['![混淆矩阵 '+s['modality']+' '+s['method']+'](../results/figures/'+s['modality']+'_'+s['method']+'_confusion.png)','']
    for path in sorted((ROOT/'results/figures').glob('*.png')):
        if '_confusion' not in path.stem:lines+=['!['+path.stem+'](../results/figures/'+path.name+')','']
    if (ROOT/'results/shortcut_audit.json').exists():
        audit=read('results/shortcut_audit.json');lines+=['### 表面属性捷径对照','',
        '| 模态 | 仅用于审计的属性 | 留出准确率 |','|---|---|---:|']
        for m,obj in audit.items():lines.append(f'| {MODALITIES[m]} | {obj["features"]} | {obj["metrics"]["accuracy"]:.4f} |')
        lines+=['','这些审计分类器未用于部署。图像格式即可高分提示数据构造存在捷径，不证明主模型直接读取了扩展名。主图像检测器只接收RGB像素，但仍可能受到历史压缩痕迹影响。','']
    lines+=['用户本人实拍／自录材料尚未提供；自主采集部分使用公开来源真实与生成测试语料，来源与哈希可追溯。补充文本来自GPT-2家族，与DistilGPT2参考模型存在模型家族亲缘，因此该测试不能代表对未知现代生成器的泛化。完整四模态跨域数据、图像／视频全量多重启攻击和对抗训练仍属于后续GPU扩展，不冒充当前已完成项。','']
    lines+=['## 11. 参考文献与来源','']+['- ['+name+']('+url+')' for name,url in REFERENCES]+['']
    text='\n'.join(lines)+'\n';(ROOT/'reports/实验报告.md').write_text(text,encoding='utf-8')
    from markdown_it import MarkdownIt
    body=MarkdownIt('commonmark').enable('table').render(text)
    (ROOT/'reports/实验报告.html').write_text('<!doctype html><meta charset="utf-8"><title>实验六报告</title><style>body{max-width:1100px;margin:40px auto;font:16px/1.7 system-ui;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:16px}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{border:1px solid #ccd2db;padding:8px}th{background:#edf3fa}img{max-width:100%}h1,h2{color:#153c66}</style>'+body,encoding='utf-8')
    dump('results/report_build.json',dict(metrics_files=len(summaries),generated=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()))
    print('Report generated with',len(summaries),'metric records',flush=True)
if __name__=='__main__':main()
