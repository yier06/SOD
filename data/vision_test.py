import os
import cv2
import json
import matplotlib.pyplot as plt

# ================= 配置区域 =================
IMAGE_DIR = 'E:/SODA-D-1/Images/Images'
JSON_FILE = 'E:/SODA-D-1/Annotations/Annotations/train.json'


# ===========================================

def visualize_existing_images(image_dir, json_file):
    # 1. 加载 JSON 数据
    print(f"正在加载标注文件: {json_file} ...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 建立映射关系
    file_to_img_info = {img['file_name']: img for img in data['images']}
    categories = {cat['id']: cat['name'] for cat in data['categories']}

    # 将标注按 "图片ID" 分组
    ann_by_img_id = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in ann_by_img_id:
            ann_by_img_id[img_id] = []
        ann_by_img_id[img_id].append(ann)

    # 3. 获取本地实际存在的图片文件
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"在 {image_dir} 中未找到任何图片文件！")
        return

    print(f"找到 {len(image_files)} 张图片，开始逐张绘制...")
    print("提示：在弹出的图片窗口中，按任意键查看下一张，按 'q' 或关闭窗口退出。")

    # 4. 遍历本地图片并逐张绘制
    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"无法读取图片: {img_path}")
            continue

        h, w, _ = img.shape

        # 尝试在 JSON 数据中查找这张图片的标注
        if img_name in file_to_img_info:
            img_info = file_to_img_info[img_name]
            img_id = img_info['id']

            # 如果这张图片有标注，就画框
            if img_id in ann_by_img_id:
                for ann in ann_by_img_id[img_id]:
                    # COCO 格式: [x, y, width, height]
                    x, y, box_w, box_h = map(int, ann['bbox'])
                    cls_id = ann['category_id']
                    cls_name = categories.get(cls_id, "Unknown")

                    # 绘制矩形框 (绿色)
                    cv2.rectangle(img, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
                    # 绘制类别名称
                    cv2.putText(img, cls_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
            else:
                print(f"提示: 图片 {img_name} 在 JSON 中没有找到对应的标注信息。")
        else:
            print(f"提示: 图片 {img_name} 不在 JSON 文件的图片列表中。")

        # --- 核心修改部分：单张显示逻辑 ---
        plt.figure(figsize=(10, 8))  # 为每张图创建一个新窗口
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(img_name)
        plt.axis('off')

        # 显示当前图片，并等待用户按键
        # waitkey(0) 表示无限等待，直到按下键盘
        plt.show()

        # 如果你想用 OpenCV 的窗口来显示（响应速度更快），可以取消下面代码的注释，并注释掉上面的 plt 部分：
        # cv2.imshow(img_name, img)
        # key = cv2.waitKey(0) # 等待按键
        # cv2.destroyAllWindows()
        # if key == ord('q'): # 按 'q' 键退出循环
        #     break


if __name__ == "__main__":
    visualize_existing_images(IMAGE_DIR, JSON_FILE)