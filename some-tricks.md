### 少量数据验证

确保网络能在少量数据集上完全拟合

### BN的设置

BN不适合做迁移学习。因为在迁移学习的时候，BN层的梯度虽然关闭了，但是running mean和running var这些统计量 还是在变化，因此在迁移学习的时候还需要把这两个统计量也冻结。或者还有一个方法，在训练完成之后，将模型设置为训练模式，从训练集抽取一个子集，只前向不反转，pytorch的train模式会自动统计running mean和running var。

### 学习率

1、warm up

2、cosine learning rate

3、swish loss

4、loss曲线抖动的厉害，增大batch size

5、训练集验证集差距大，增大正则项

### 调参

1、随机网格搜索

2、贝叶斯优化

3、用大模型+early stop往往效果较好



