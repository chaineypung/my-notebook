### 任务：对细胞轮廓的分割

### 数据处理：

1、patch(sliding window(加入少量的overlap)，random crop)

2、水平，竖直翻转，随机旋转，放大，缩小，镜面反射

3、模糊，锐化，加性，乘性噪声

4、遮挡(random erasing)

### 模型：

1、考虑Deeplab和U-Net的参数过多，这里选择LinkNet

2、CE+FOCAL+DICE的loss

### TRICKS：

1、两阶段法，将第一级网络的粗分割输入网络的第二级做精细分割

### 参考链接：

[U-RISC 神经元识别大赛第一名解决方案 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/114479422?utm_source=qq&utm_medium=social&utm_oi=800716886307389440)