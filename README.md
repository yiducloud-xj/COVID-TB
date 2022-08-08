# COVID-TB
本github项目是文章“基于深度学习的流调报告事件抽取研究”对应的运行代码

## 1. 系统需求

### 软件版本:

+ Python version 3.7.0 

### Python库:
+ torch==1.4.0

+ sklearn == 1.0.1

+ pandas == 1.2.4

+ numpy == 1.19.2

+ torchmetrics == 0.9.2

+ transformers==4.20.1

### 其他非标准化硬件需求:
模型训练比较依赖GPU，若以CPU运算能力需要大于五天的训练时间（6个epoch）

## 2. 安装说明
可执行程序，无安装需要，GPU相关驱动和CUDA版本请参照NVIDIA官网: https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux

需要原始的BERT-BASE-CHINESE预训练模型，放置于代码根目录下，来源于huggingface的原始pytorch版本: https://huggingface.co/bert-base-chinese


## 3. 执行步骤
1，运行main.py以执行模型训练和预测流程（单次执行即可完成训练和预测流程），于代码16-20行设定CPU/GPU模式以及具体执行的显卡序号。同时要修改datasets/ECR_COVID_19/load_datasets.py中11-15行，以及models/je_model.py中11-15行中的GPU/CPU设定。

2，如果GPU显存较小，请手动调小main.py中31行的BATCH_SIZE值，同时在datasets/ECR_COVID_19/load_datasets.py中也需要进行一致的修改。
