from PIL import Image
from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()

    # mode = predict / dir_predict
    mode = "dir_predict"
    dir_origin_path = 'pdf_img'
    dir_save_path = "pdf_img_out"

    if mode == "predict":
        while True:
            img = input('Please input image file path:')
            try:
                image = Image.open(img)
            except:
                print(f'Path:[ {img} ] Open Error! Please Check!')
                continue
            else:
                pre_image, bboxes = yolo.detect_image(image)
                for box in bboxes:
                    print(box)
                pre_image.show()

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                pre_image, bboxes = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                pre_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    else:
        raise AssertionError(
            " Please check mode: 'predict', 'dir_predict'! ")
