# semantic_segmentation_based_on_deeplab_v3_using_facade_dataset
semantic segmentation based on deeplab v3 using facade dataset

代码参考来源：[pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)

# facade数据集预处理
使用以下代码创建标准数据集：

    import os
    import numpy as np
    root = r"F:\chrome\facade\JPEGImages"
    output = r"F:\chrome\facade\ImageSets\Segmentation"
    filename = []
    #从存放原图的目录中遍历所有图像文件
    for root, dir, files in os.walk(root):
        for file in files:
            # print(file)
            filename.append(file[:-4])  # 去除后缀，存储
    
    
    #打乱文件名列表
    np.random.shuffle(filename)
    #划分训练集、测试集, 默认比例6:2:2
    train = filename[:int(len(filename)*0.6)]
    trainval = filename[int(len(filename)*0.6):int(len(filename)*0.8)]
    val = filename[int(len(filename)*0.8):]
    
    #分别写入train.txt, test.txt
    with open(os.path.join(output,'train.txt'), 'w') as f1, open(os.path.join(output,'trainval.txt'), 'w') as f2,open(os.path.join(output,'val.txt'), 'w') as f3:
        for i in train:
            f1.write(i + '\n')
        for i in trainval:
            f2.write(i + '\n')
        for i in val:
            f3.write(i + '\n')
    
    print('成功！')

标准数据集结果如下：
    -facade(数据集名称)
    	- ImageSets
    		- Segmentation
    			- train.txt
    			- test.txt
    			- val.txt
    	- JPEGImages(原始图片)
    	- SegmentationClass(分割后图片)
 
# 使用自己的数据集训练deeplab v3模型
- mypath.py 中加入自己数据集的路径
- 复制任意一份`dataset_name.py`文件，并重命名为自己的数据集名称（以下以facade数据集为例）
  - 更改init函数：base_dir=Path.db_root_dir('facade')
  - 更改__str__(self)函数：return 'facede(split=' + str(self.split) + ')'
- 修改dateloaders目录下`utils.py`
  - 添加自己数据集的函数，例如`get_facade_labels()`
  - 按照类别增加颜色数组
  - 在`decode_segmap`函数内添加代码，其中`n_classes`是分割的类别数
- 在dataloaders目录下修改`__init__.py`
  - import facade
  - 按其它数据集的格式，增加facade数据集处理代码
- 在同级目录中修改train.py，添加自己数据集的名称

# 使用以下代码训练，按需求更改参数

      python train.py --backbone mobilenet --lr 0.007 --workers 1 --epochs 50 --batch-size 8 --gpu-ids 0 --checkname deeplab-mobilenet

学习自：https://blog.csdn.net/sazass/article/details/127262441
