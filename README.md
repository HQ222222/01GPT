
<h1 align="center"> ScratchLLMStepByStep</h1>

欢迎来到这套全面的从零开始编写并训练大语言模型的教程！本项目旨在为对语言模型和深度学习感兴趣的开发者提供一套系统的、易于理解的学习资源。通过本系列教程，您将逐步了解并掌握大语言模型的基本概念、核心算法及其实现细节。

本教程将会带你从分词器训练开始，一步一步编写和实现自己的attention、transformer以及gptmodel，并对这个模型进行预训练、监督微调(SFT)、偏好学习（DPO），最终训练出一个可以进行对话聊天的大语言模型。

## 💥 目标受众

本教程适合具有以下背景的读者：
- 具备基本的编程知识，尤其是Python
- 对机器学习和深度学习有一定的了解
- 希望深入理解语言模型的工作原理和实现方法

## 💥 章节结构

- [带你从零训练tokenizer](./tokenizer.ipynb)
- [词嵌入和位置嵌入](./embedding.ipynb)
- [从零认识自注意力](./attention.ipynb)
- [实现因果注意力机制](./causal_attention.ipynb)
- [实现多头注意力](./multi_attention.ipynb)
- [构建TransformerBlock](./transformer.ipynb)
- ……(持续更新中)

## 💥 数据集
相关训练所需数据集的下载地址。
- [分词器训练数据集](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)
- [预训练数据集](http://share.mobvoi.com:5000/sharing/O91blwPkY)
- [SFT数据集](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data/resolve/master/sft_data_zh.jsonl)
- [DPO数据集](https://huggingface.co/datasets/Skepsun/huozi_rlhf_data_json)

## 💥 运行环境

仅是我个人的软硬件环境配置，自行酌情更改：

* Ubuntu == 18.04
* Python == 3.10
* Pytorch == 2.4.0
* CUDA == 12.1

前面编写模型结构的部分对GPU不是强依赖，后面预训练、SFT、DPO需要使用GPU进行训练，并且尽量是多块GPU（个人使用的4块24G的GPU进行训练）。

有两篇配套的环境搭建教程可以作为参考：
- [conda&pytorch环境搭建笔记](https://golfxiao.blog.csdn.net/article/details/140819506)
- [cuda安装笔记](https://golfxiao.blog.csdn.net/article/details/140877932)

## 💥 如何开始？
1. 克隆本项目到本地：
```
git clone https://github.com/golfxiao/ScratchLLMStepByStep.git
```
2. 下载上面列出的依赖数据集，将notebook中用到的数据集地址修改成你本地地址
3. 按照顺序阅读每个Notebook，并运行其中的代码。
4. 根据需要修改和实验代码，以加深对相关概念的理解。


最后，感谢您阅读这个教程。如果觉得对您有所帮助，可以考虑请我喝杯咖啡作为鼓励😊

![a cup of tea](./img/cup_of_tea.jpg)