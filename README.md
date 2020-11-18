## NLP

#### On the Sentence Embeddings from Pre-trained Language Models
使用glow解决预训练语言模型语义区分能力差 ，无需调整模型参数，只需对模型输出的特征加一个glow

### ON THE STABILITY OF FINE-TUNING BERT: MISCONCEPTIONS,EXPLANATIONS, AND STRONG BASELINES
探索bert在微调时为什么为不稳定，主要原因是梯度遗忘和数据集太小。解决方法是：1、使用更小的学习率以及bias更正；2、延长迭代次数，直到训练loss接近0。另外对不同的模型使用了不同的超参数，比如albert就不需要dropout
