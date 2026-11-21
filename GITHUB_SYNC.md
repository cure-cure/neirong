# GitHub与AutoDL同步说明

仓库：https://github.com/cure-cure/neirong

这是实验六的本地代码基线。用户已在AutoDL执行过实验，但服务器侧新增代码尚未合入此提交；本次提交不代表那些GPU更新已经被审查或同步。

## 仓库范围

包含四模态源码、UI、实验脚本、测试、配置、第三方代码与许可证、课程PPT、依赖文件、数据划分清单与交接说明。排除Windows虚拟环境、原始数据、特征缓存、模型权重、运行结果、上传ZIP和临时文件。

服务器已有数据和模型应原样保留。Git不是原始数据或实验结果备份；源代码仓库与单独的资产包、结果包共同构成完整可复现交付。

## AutoDL已修改代码：推荐的接入方式

先在服务器确认当前工作目录，并将已有源码、配置、依赖、新权重和结果做版本化备份。不要在已有GPU工作目录直接执行强制checkout、reset --hard、clean或覆盖式解压。

在旁边克隆仓库，建立更新分支：

```bash
cd /root/autodl-tmp
git clone https://github.com/cure-cure/neirong.git neirong-git
cd neirong-git
git switch -c autodl/experiment6-update
```

比较服务器当前代码与克隆目录的文件差异，然后把确认过的更新复制到克隆目录。例如更新的src、scripts、tests、app.py和相关配置。不要把整个虚拟环境、数据、models或results递归复制进仓库。若有新增方法，同时合并依赖、加载路径、校准、选择规则、测试和说明。

代码准备好后：

```bash
git status --short
git diff --stat
git diff
git add src scripts tests configs app.py
# 有依赖或说明变更时，再显式添加对应文件。
git diff --cached --stat
git diff --cached --check
git commit -m "Update experiment 6 from AutoDL"
git push -u origin autodl/experiment6-update
```

这些命令中的git add路径只是常见示例，应以实际修改为准。推送前检查敏感信息与大文件。通过GitHub比较更新分支与main，确认后再合并；不要强制推送覆盖本地基线。

## 后续常规同步

在工作树干净时执行 `git pull --ff-only`。若本地与远程都有更新，先提交或备份当前改动，在更新分支上解决冲突，保留双方实现和有效实验，不用强制覆盖掩盖冲突。

GitHub认证使用自己的账号授权或安全凭据存储。不要把token、密码、SSH私钥写入源码、说明或日志。

完整课程要求与实验交接材料保留在本机和服务器原工作目录；已完成的新GPU工作应由服务器侧据实际结果更新说明，不能重新把所有任务都当作未执行。
