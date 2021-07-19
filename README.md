# 基于pytorch+bert的中文文本分类

本项目是基于pytorch+bert的中文文本分类，使用的数据集是THUCNews，数据地址：<a href="https://github.com/gaussic/text-classification-cnn-rnn">THUCNews</a>，在该项目中，数据存放在data/cnews/raw_data文件夹下。<br>

## 使用依赖
```python
torch==1.6.0
transformers==4.5.1
```
## 相关说明
--logs：存放日志<br>
--checkpoints：存放保存的模型<br>
--data：存放数据<br>
--utils：存放辅助函数<br>
--bert_config.py：相关配置<br>
--dataset.py：制作数据为torch所需的格式<br>
--preprocess.py：数据预处理成bert所需要的格式<br>
--models.py：存放模型代码
--main.py：主运行程序，包含训练、验证、测试、预测以及相关评价指标的计算<br>
要预先下载好预训练的bert模型，放在和该项目同级下的model_hub文件夹下，即：<br>
model_hub/bert-base-chinese/
相关下载地址：<a href="https://huggingface.co/bert-base-chinese/tree/main=">bert-base-chinese</a><br>
需要的是vocab.txt、config.json、pytorch_model.bin

## 一般步骤
先从preprocess.py中看起，里面有处理数据为bert所需格式的相关代码，相关运行结果会保存在logs下面的preprocess.log中。然后看dataset.py代码，里面就是制作成torch所需格式的数据集，我们需要运行该文件将数据存储为pickle文件。感兴趣的可以继续看看models.py中模型建立的过程。最终的运行主函数在main.py中。在main.py中运行的结果会保存在logs下的main.log中。

## 运行
```python
python main.py \
--bert_dir="../model_hub/bert-base-chinese/" \
--data_dir="./data/final_data/" \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=10 \
--seed=123 \
--gpu_ids="0" \
--max_seq_len=256 \
--lr=3e-5 \
--other_lr=3e-4 \
--train_batch_size=32 \
--train_epochs=5 \
--eval_batch_size=32 \
```
在main.py中有相关代码控制训练、验证、测试、预测，根据需要注释掉其他的。

## 结果
### 训练和验证（部分结果）
```
2021-07-19 15:30:35,608 - INFO - main.py - train - 73 - 【train】 epoch：0 step:399/7815 loss：0.078945
2021-07-19 15:31:13,990 - INFO - main.py - train - 79 - 【dev】 loss：20.292041 accuracy：0.9666 micro_f1：0.9666 macro_f1：0.9663
2021-07-19 15:31:13,991 - INFO - main.py - train - 81 - ------------>保存当前最好的模型
```
### 测试
```python
========进行测试========
2021-07-19 15:49:16,818 - INFO - main.py - <module> - 226 - 【test】 loss：18.493065 accuracy：0.9708 micro_f1：0.9708 macro_f1：0.9706
2021-07-19 15:49:16,834 - INFO - main.py - <module> - 228 -               precision    recall  f1-score   support

          教育       0.97      0.97      0.97       500
          娱乐       1.00      0.95      0.97       500
          家居       0.99      0.87      0.93       500
          房产       0.92      0.97      0.94       500
          科技       0.97      1.00      0.99       500
          时尚       0.95      1.00      0.98       500
          体育       1.00      1.00      1.00       500
          财经       0.95      1.00      0.97       500
          时政       0.98      0.96      0.97       500
          游戏       0.99      0.99      0.99       500

    accuracy                           0.97      5000
   macro avg       0.97      0.97      0.97      5000
weighted avg       0.97      0.97      0.97      5000
```
## 预测
预测的结果没有保存在log中，在main.py中最下面一段代码就是的，自行运行即可。



