#### On the Sentence Embeddings from Pre-trained Language Models
使用glow解决预训练语言模型语义区分能力差 ，无需调整模型参数，只需对模型输出的特征加一个glow

#### ON THE STABILITY OF FINE-TUNING BERT: MISCONCEPTIONS,EXPLANATIONS, AND STRONG BASELINES
探索bert在微调时为什么为不稳定，主要原因是梯度遗忘和数据集太小。解决方法是：1、使用更小的学习率以及bias更正；2、延长迭代次数，直到训练loss接近0。另外对不同的模型使用了不同的超参数，比如albert就不需要dropout

#### Freelb: Enhanced adversarial training for natural language understanding.
#### SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization
提升微调任务效果

#### DELIGHT: DEEP AND LIGHT-WEIGHT TRANSFORMER
轻量化transformer模块可降低2到3倍的参数量

#### The Depth-to-Width Interplay in Self-Attention
研究transformer 中层数和宽度的关系，给出了固定模型大小下层数和宽度的建议

#### Unsupervised Data Augmentationfor Consistency Training
uda 无监督数据增广。两个loss，一个是有标签的loss；一个是无标签的loss，无标签数据会先进行增广，然后分别过模型得到prob输出，计算KL loss，同时对没有增广的prob进行梯度截断。

#### MINILM: Deep Self-Attention Distillation forTask-Agnostic Compression of Pre-Trained Transformers
知识蒸馏：对self attention score和value-value的attention score进行蒸馏；DistillBERT是对prob和embedding蒸馏；TinyBERT是对embedding、多层hidden state、self attention score蒸馏；MobileBERT 是对prob、多层hidden state和self attention score蒸馏

#### ConvBERT: Improving BERT with Span-basedDynamic Convolution
看起来并行性不是很友好，不适合加速优化

#### Pre-training via Paraphrasing
判断某一query和相关模型召回的一系列doc是否是匹配的

#### PatchBERT: Just-in-Time, Out-of-Vocabulary Patching
解决预训练好的模型，在下游任务出现oov的情况


#### BERTRAM: Improved Word Embeddings Have Big Impact onContextualized Model Performance
解决长尾词embedding较差的问题，首先收集包含某一个长尾词的文本，将这些文本一起输入bert，然后利用这些文本的cls token特征组合出一个新的该长尾词的embedding，并将该embedding增加到bert的原始embedding进去。效果提升较明显。


#### TABERT: Pretraining for Joint Understanding ofTextual and Tabular Data
针对表格的预训练方法

#### Cross-Thought for Sentence Encoder Pre-training
提升bert的句子表征能力，mask的字通过其周围的句子来预测

#### MacBERT：MLM as correction BERT
1、掩蔽1个到4个词，概率为0.4，0.3，0.2，0.1；2、15%的词做处理，其中80%做同义词替换（来自w2v），10% mask，10%不变。3、SOP任务


#### UPRec: User-Aware Pre-training forRecommender Systems
推荐里面的预训练，输入是用户行为序列，mask对象是用户点击的商品。预测mask，用户属性和社交关系

#### Survey on graph embeddings and their applications to machine learning problems on graphs
图embedding应用综述

#### SLM: Learning a Discourse Language Representation with SentenceUnshuffling
句子表征学习，句子顺序还原任务

#### Revisiting ResNets: Improved Training and Scaling Strategies
利用各种技巧提升resnet的训练效果


#### Keyword-Attentive Deep Semantic Matching
加入keyword信息，加入的方式，对于表征模型值得尝试

#### Dependency Parsing based Semantic Representation Learning with Graph Neural Network for Enhancing Expressiveness of Text-to-Speech
表征模型加入句法分析，在bert后面加入GNN模型融合句法分析

#### An Empirical Study of Training Self-Supervised Vision Transformers
moco v3 解决大batch vit训练不稳定问题

#### SimCSE: Simple Contrastive Learning of Sentence Embeddings
利用对比学习，学习句子表征。提出了监督和无监督两种方法，其中无监督用的是dropout进行文本扩充，比其他文本扩充方式效果更好，感觉挺有用。

#### Search-oriented Differentiable Product Quantization
可学习pq量化，里面对比了多种pq效果

#### Extract then Distill: Efficient and EffectiveTask-Agnostic BERT Distillation
根据神经元重要度进行宽度降维，然后进行【0 n/m 2n/m ... n】层对应蒸馏。

#### DefSent: Sentence Embeddings using Definition Sentences
用词典的词定义句预测被定义的词，挺有意思的一个思路，值得尝试


#### ConSERT: A Contrastive Framework for Self-Supervised SentenceRepresentation Transfer
也是对比学习，学习句子表征，效果不如SimCSE. 扩充方式包括了对抗攻击，token丢弃，输入特征丢弃（128维，丢弃其中几维），位置编码shuffle，输入特征dropout
