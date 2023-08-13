import colorsys
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.yolo import YoloBody
from utils.utils import cvtColor, get_anchors, get_classes, preprocess_input, resize_image, show_config
from utils.utils_bbox import DecodeBox


class YOLO(object):
    def __init__(self,
                 model_path='config/best_epoch_weights.pth',
                 classes_path='config/math_classes.txt',
                 anchors_path='config/yolo_anchors.txt',
                 anchors_mask=([6, 7, 8], [3, 4, 5], [0, 1, 2]),
                 input_shape=(416, 416),
                 confidence=0.5,
                 nms_iou=0.3,
                 letterbox_image=False,
                 cuda=True
                 ):

        self.model_path = model_path
        self.classes_path = classes_path
        self.anchors_path = anchors_path
        self.anchors_mask = anchors_mask
        self.input_shape = input_shape
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.letterbox_image = letterbox_image
        self.cuda = cuda

        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
                                   self.anchors_mask)
        self.net = self.generate()

        # 画框设置颜色
        if self.num_classes == 2:
            self.colors = [(255, 0, 0), (0, 0, 255)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        show_config(**{
            "model_path": model_path,
            "classes_path": classes_path,
            "anchors_path": anchors_path,
            "anchors_mask": anchors_mask,
            "input_shape": input_shape,
            "confidence": confidence,
            "nms_iou": nms_iou,
            "letterbox_image": letterbox_image,
            "cuda": cuda
        })

    # 载入预训练好的模型
    def generate(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        net = YoloBody(self.anchors_mask, self.num_classes)
        net.load_state_dict(torch.load(self.model_path, map_location=device))
        net = net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        if self.cuda:
            net = nn.DataParallel(net)
            net = net.cuda()
        return net

    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        # 保证输入都为 RGB 防止灰度图等
        image = cvtColor(image)
        # resize 416 * 416
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # 添加 batch_size 维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)
            if results[0] is None:
                return image, []

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        # 设置字体与边框厚度
        font = ImageFont.truetype(font='config/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[0] // 2).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape) // 2, 1))

        # 绘制图像
        boxes = []
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box
            boxes.append((predicted_class, left, top, right, bottom))

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            # print(label, top, left, bottom, right)
            # top, left, bottom, right

            # 是否需要打印类别名字，如需打印，请去掉注释
            label_size = draw.textsize(label, font)
            # label = label.encode('utf-8')
            # if top - label_size[1] >= 0:
            #     text_origin = np.array([left, top - label_size[1]])
            # else:
            #     text_origin = np.array([left, top + 1])

            for t in range(thickness):
                draw.rectangle((left + t, top + t, right - t, bottom - t), outline=self.colors[c])
            # draw.rectangle((tuple(text_origin), tuple(text_origin + label_size)), fill=self.colors[c])
            # draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image, boxes
