### BN的设置

最新的论文指出，BN不适合做迁移学习。因为在迁移学习的时候，BN层的梯度虽然关闭了，但是running mean和running var这些统计量 还是在变化，因此在迁移学习的时候还需要把这两个统计量也冻结。或者还有一个方法，在训练完成之后，将模型设置为训练模式，从训练集抽取一个子集，只前向不反转，pytorch的train模式会自动统计running mean和running var。

cosine learning rate

swish loss

