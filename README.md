# 研究天气与车位识别的关系

```
🕙 2023-11-18
```
 
**文件说明**
1. checkpoint 保存训练好的模型
2. data 保存图片和标签数据 
3. datasets 根据data制作的数据集 
4. networks 网络模型结构 
5. result  训练结果  
6. utils  生成不同天气的脚本函数
7. config.py 配置脚本参数
8. generate_weather_img.py 生成天气图像
9. script.py 自动化运行命令脚本
10. train.py 训练

**初次使用步骤**
1. 将原始图像放到 data 下的 original 文件夹中，原始标签放到 data下的original_labels.txt中
2. 运行generate_weather_img.py生成数据集
3. 设置需要的脚本参数，运行tain.py或者script.py训练需要的模型


