# 数学公式检测

> 北京航空航天大学
>
> 张杰宣

## 简单使用

打开 predict.py 即可使用，其中 predict.py 的 mode 有 :

- predict 单张图片预测，可以直接使用，无需更改任何内容

- dir_predict 对一个文件夹下所有文件预测，并输出预测图片到指定文件夹，在使用时，请注意更改文件夹的 Path 路径 :

  ```python
  dir_origin_path = 'pdf_img'
  dir_save_path = "pdf_img_out"
  ```

## 类别以及框的颜色

我们在本次检测任务中设置了两个类别：

- display 对应红色框，只包含公式本身，不包含序号
- display_all 对应蓝色框，包含公式以及对应的序号

具体的区别就是是否含有编号，如下图 :

<img src=".\source\1.png" alt="image-20230813163946494" style="zoom:50%;" />

## 测试效果

<img src=".\pdf_img_out\1.png" alt="1" style="zoom:50%;" />

我们在默认文件夹下已经放置了几张图片，并进行了处理，想观察直接效果可以观看 pdf_img_out 文件夹内容。

## 如何拿到预测框的数值?

我们在 Inference 的过程中，同时将预测框也进行了输出，如代码中的:

```python
pre_image, bboxes = yolo.detect_image(image)
```

bboxes 存储着该张图片的所有预测框信息，格式为 list[(x_min, y_min, x_max, y_max)]，其中 list 的长度为该张图片拥有多少个预测框。

