# machine_trans
使用seq2seq模型进行机器翻译
把英语翻译成汉语（可通过替换训练语料用作其他使用seq2seq模型的功能）。
运行方式： 第一步：python data_utils.py 处理语料。
          
          第二步：python train.py  训练模型
          
          第三步：python inference.py 对训练好的模型进行推理验证



测试结果如图：


![Image text](https://raw.githubusercontent.com/wdwcn/machine_trans/master/pics/inference.png)


补充：seq2seq.ini 为模型的配置文件。

