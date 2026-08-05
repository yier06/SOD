import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Optional

import cv2


# ============================================================
# 1. 数据集路径配置
# ============================================================

# 所有图片解压后的总目录。
#
# 例如：
# D:\SODA-D\Images
#
# 目录内部可以是：
# Images/
# ├── 000001.jpg
# ├── 000002.jpg
# └── ...
#
# 也可以存在多层目录：
# Images/
# └── SODA-D/
#     └── Images/
#         ├── 000001.jpg
#         └── ...
IMAGE_ROOT = Path(r"F:\SODA-D\Images\Images")


# ------------------------------------------------------------
# annotation 的使用方式
# ------------------------------------------------------------

# 方式一：annotation 已经解压为目录
ANNOTATION_ROOT = Path(r"F:\SODA-D\Annotations\Annotations")

# # 方式二：annotation 仍然是 ZIP
# ANNOTATION_ZIP_PATH = Path(r"D:\SODA-D\Annotations.zip")

# # True：从 Annotations.zip 中读取 JSON
# # False：从 ANNOTATION_ROOT 目录读取 JSON
READ_ANNOTATION_FROM_ZIP = False


# ------------------------------------------------------------
# 当前需要查看的数据划分
# ------------------------------------------------------------

# 可选：
# "train"
# "val"
# "test"
SPLIT = "train"


# ------------------------------------------------------------
# JSON 文件名配置
# ------------------------------------------------------------

# 如果你的文件名明确，可以直接填写。
#
# 例如：
# "train.json"
# "val.json"
# "test.json"
#
# 如果设为 None，程序会在目录或 ZIP 中自动寻找包含
# train、val、test 关键词的 JSON。
ANNOTATION_JSON_NAMES = {
    "train": None,
    "val": None,
    "test": None,
}


# ============================================================
# 2. 显示配置
# ============================================================

WINDOW_NAME = "SODA-D COCO Visualization"

MAX_SHOW_WIDTH = 1600
MAX_SHOW_HEIGHT = 900

BOX_THICKNESS = 2
FONT_SCALE = 0.5

SHOW_CATEGORY_NAME = True
SHOW_ANNOTATION_ID = False
SHOW_IMAGE_ID = False
SHOW_TOP_INFORMATION = True

# 是否忽略特殊标注
SKIP_CROWD = False
SKIP_IGNORE = False

# 最小框面积，0 表示不筛选
MIN_BBOX_AREA = 0.0

# 是否只显示包含标注框的图片
#
# test 没有公开标注时必须设为 False
ONLY_SHOW_IMAGES_WITH_ANNOTATIONS = False

# 是否跳过 JSON 中找不到实际文件的图片
SKIP_MISSING_IMAGES = True

# 类别颜色随机种子
COLOR_SEED = 42


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# 3. annotation 读取
# ============================================================

def normalize_path(path: str) -> str:
    """
    将 Windows 和 ZIP 内路径统一成 POSIX 格式。
    """
    return str(path).replace("\\", "/").lstrip("./")


def score_json_name(json_name: str, split: str) -> int:
    """
    给候选 JSON 文件打分。

    当前 split 关键词的权重最高。
    """
    lower_name = json_name.lower()
    score = 0

    if split.lower() in lower_name:
        score += 100

    if "instance" in lower_name:
        score += 20

    if "annotation" in lower_name:
        score += 10

    if "coco" in lower_name:
        score += 5

    if "soda" in lower_name:
        score += 2

    # 避免 train 查到 val/test
    for other_split in {"train", "val", "test"} - {split}:
        if other_split in lower_name:
            score -= 100

    return score


def select_json_name(
    json_names: list[str],
    split: str,
    specified_name: Optional[str],
) -> str:
    """
    从候选 JSON 中选择当前划分文件。
    """
    if not json_names:
        raise FileNotFoundError("没有找到任何 JSON 文件。")

    normalized_to_original = {
        normalize_path(name): name
        for name in json_names
    }

    if specified_name is not None:
        normalized_specified = normalize_path(specified_name)

        if normalized_specified in normalized_to_original:
            return normalized_to_original[normalized_specified]

        # 允许只填写文件名
        basename_matches = [
            name
            for name in json_names
            if PurePosixPath(normalize_path(name)).name
            == PurePosixPath(normalized_specified).name
        ]

        if len(basename_matches) == 1:
            return basename_matches[0]

        raise FileNotFoundError(
            f"没有找到指定 JSON：{specified_name}"
        )

    ranked_names = sorted(
        json_names,
        key=lambda name: (
            score_json_name(name, split),
            -len(name),
        ),
        reverse=True,
    )

    selected_name = ranked_names[0]
    selected_score = score_json_name(selected_name, split)

    if selected_score <= 0:
        print("\n发现的 JSON：")

        for name in json_names:
            print(f"  {name}")

        raise RuntimeError(
            f"无法可靠地自动找到 {split} JSON。\n"
            "请在 ANNOTATION_JSON_NAMES 中明确填写文件名。"
        )

    print(f"自动选择 {split} JSON：{selected_name}")

    return selected_name


def load_coco_from_directory(
    annotation_root: Path,
    split: str,
    specified_name: Optional[str],
) -> tuple[dict, str]:
    """
    从已经解压的 annotation 目录读取 JSON。
    """
    if not annotation_root.exists():
        raise FileNotFoundError(
            f"annotation 目录不存在：{annotation_root}"
        )

    json_paths = list(annotation_root.rglob("*.json"))

    selected_name = select_json_name(
        json_names=[
            normalize_path(path.relative_to(annotation_root))
            for path in json_paths
        ],
        split=split,
        specified_name=specified_name,
    )

    selected_path = annotation_root / Path(selected_name)

    with selected_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        coco_data = json.load(file)

    return coco_data, str(selected_path)


def load_coco_from_zip(
    annotation_zip_path: Path,
    split: str,
    specified_name: Optional[str],
) -> tuple[dict, str]:
    """
    从 annotation ZIP 内直接读取 JSON，不解压到磁盘。
    """
    if not annotation_zip_path.exists():
        raise FileNotFoundError(
            f"annotation ZIP 不存在：{annotation_zip_path}"
        )

    with zipfile.ZipFile(annotation_zip_path, "r") as zip_file:
        json_names = [
            name
            for name in zip_file.namelist()
            if not name.endswith("/")
            and name.lower().endswith(".json")
        ]

        selected_name = select_json_name(
            json_names=json_names,
            split=split,
            specified_name=specified_name,
        )

        json_bytes = zip_file.read(selected_name)

    coco_data = json.loads(
        json_bytes.decode("utf-8-sig")
    )

    return coco_data, selected_name


def load_selected_coco() -> tuple[dict, str]:
    """
    根据配置读取当前 split 对应的 COCO JSON。
    """
    split = SPLIT.lower()

    if split not in {"train", "val", "test"}:
        raise ValueError(
            f"SPLIT 必须是 train、val 或 test，当前为：{SPLIT}"
        )

    specified_name = ANNOTATION_JSON_NAMES.get(split)

    if READ_ANNOTATION_FROM_ZIP:
        return load_coco_from_zip(
            annotation_zip_path=ANNOTATION_ZIP_PATH,
            split=split,
            specified_name=specified_name,
        )

    return load_coco_from_directory(
        annotation_root=ANNOTATION_ROOT,
        split=split,
        specified_name=specified_name,
    )


def validate_coco_data(coco_data: dict) -> None:
    """
    检查基本 COCO 格式。

    test 可能没有 annotations，所以不强制要求该字段。
    """
    required_keys = {
        "images",
        "categories",
    }

    missing_keys = required_keys - set(coco_data.keys())

    if missing_keys:
        raise ValueError(
            f"JSON 缺少 COCO 必需字段：{sorted(missing_keys)}"
        )

    if not isinstance(coco_data["images"], list):
        raise TypeError("images 字段必须是列表。")

    if not isinstance(coco_data["categories"], list):
        raise TypeError("categories 字段必须是列表。")

    if "annotations" in coco_data:
        if not isinstance(coco_data["annotations"], list):
            raise TypeError("annotations 字段必须是列表。")


# ============================================================
# 4. 构建 COCO 关联索引
# ============================================================

def build_coco_indexes(coco_data: dict):
    """
    COCO 关联关系：

    images[].id
        ↑
    annotations[].image_id

    images[].file_name
        ↓
    实际图像文件
    """
    image_infos = sorted(
        coco_data["images"],
        key=lambda item: item.get("id", 0),
    )

    category_id_to_name = {
        category["id"]: category["name"]
        for category in coco_data["categories"]
    }

    image_id_to_annotations = defaultdict(list)

    # test JSON 可能没有 annotations
    annotations = coco_data.get("annotations", [])

    for annotation in annotations:
        image_id = annotation.get("image_id")

        if image_id is not None:
            image_id_to_annotations[image_id].append(annotation)

    return (
        image_infos,
        category_id_to_name,
        image_id_to_annotations,
    )


def create_category_colors(
    category_ids,
    seed: int = 42,
) -> dict[int, tuple[int, int, int]]:
    """
    为每个类别创建固定 BGR 颜色。
    """
    random_generator = random.Random(seed)

    colors = {}

    for category_id in sorted(category_ids):
        colors[category_id] = (
            random_generator.randint(50, 255),
            random_generator.randint(50, 255),
            random_generator.randint(50, 255),
        )

    return colors


# ============================================================
# 5. 建立实际图片索引
# ============================================================

def build_image_index(image_root: Path):
    """
    对统一 Images 目录递归建立索引。

    支持 JSON 中 file_name 是：
        000001.jpg
        train/000001.jpg
        images/000001.jpg

    同时支持实际图片位于多层目录中。
    """
    if not image_root.exists():
        raise FileNotFoundError(
            f"图片根目录不存在：{image_root}"
        )

    exact_index: dict[str, Path] = {}
    basename_index: dict[str, list[Path]] = defaultdict(list)
    suffix_index: dict[str, list[Path]] = defaultdict(list)

    print(f"正在扫描图片目录：{image_root}")

    for image_path in image_root.rglob("*"):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_name = normalize_path(
            image_path.relative_to(image_root)
        )

        exact_index[relative_name] = image_path
        basename_index[image_path.name].append(image_path)

        parts = PurePosixPath(relative_name).parts

        for start_index in range(len(parts)):
            suffix_name = "/".join(parts[start_index:])
            suffix_index[suffix_name].append(image_path)

    print(f"扫描到 {len(exact_index)} 张图片。")

    return exact_index, basename_index, suffix_index


def find_image_path(
    coco_file_name: str,
    exact_index: dict[str, Path],
    basename_index: dict[str, list[Path]],
    suffix_index: dict[str, list[Path]],
) -> Optional[Path]:
    """
    根据 JSON 中的 file_name 查找实际图片。
    """
    normalized_name = normalize_path(coco_file_name)

    # 1. 相对路径完全一致
    if normalized_name in exact_index:
        return exact_index[normalized_name]

    # 2. 实际路径以后缀形式匹配 JSON 路径
    suffix_matches = suffix_index.get(
        normalized_name,
        [],
    )

    if len(suffix_matches) == 1:
        return suffix_matches[0]

    if len(suffix_matches) > 1:
        return min(
            suffix_matches,
            key=lambda path: len(str(path)),
        )

    # 3. 只按文件名匹配
    basename = PurePosixPath(normalized_name).name
    basename_matches = basename_index.get(
        basename,
        [],
    )

    if len(basename_matches) == 1:
        return basename_matches[0]

    if len(basename_matches) > 1:
        print(
            f"[警告] 存在多个同名图片：{basename}\n"
            f"候选数量：{len(basename_matches)}，"
            "暂时选择路径最短的一个。"
        )

        return min(
            basename_matches,
            key=lambda path: len(str(path)),
        )

    return None


# ============================================================
# 6. 绘制标注
# ============================================================

def clip_coco_bbox(
    bbox,
    image_width: int,
    image_height: int,
):
    """
    COCO bbox：

        [x, y, width, height]

    转为：

        [x1, y1, x2, y2]
    """
    if bbox is None or len(bbox) != 4:
        return None

    try:
        x, y, width, height = map(float, bbox)
    except (TypeError, ValueError):
        return None

    if width <= 0 or height <= 0:
        return None

    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + width))
    y2 = int(round(y + height))

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def get_text_color(
    background_color: tuple[int, int, int],
) -> tuple[int, int, int]:
    """
    根据标签背景亮度选择黑色或白色字体。
    """
    b, g, r = background_color

    brightness = (
        0.114 * b
        + 0.587 * g
        + 0.299 * r
    )

    if brightness > 150:
        return 0, 0, 0

    return 255, 255, 255


def draw_label(
    image,
    text: str,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    """
    绘制类别名称背景和文字。
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = 1
    padding = 3

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        FONT_SCALE,
        text_thickness,
    )

    label_height = (
        text_height
        + baseline
        + padding * 2
    )

    # 默认标签位于框上方
    label_top = y1 - label_height
    label_bottom = y1

    # 上方空间不足时放到框内部
    if label_top < 0:
        label_top = y1
        label_bottom = min(
            image.shape[0] - 1,
            y1 + label_height,
        )

    label_left = max(0, x1)

    label_right = min(
        image.shape[1] - 1,
        label_left + text_width + padding * 2,
    )

    cv2.rectangle(
        image,
        (label_left, label_top),
        (label_right, label_bottom),
        color,
        thickness=-1,
    )

    cv2.putText(
        image,
        text,
        (
            label_left + padding,
            label_bottom - baseline - padding,
        ),
        font,
        FONT_SCALE,
        get_text_color(color),
        text_thickness,
        cv2.LINE_AA,
    )


def draw_annotations(
    image,
    annotations: list[dict],
    category_id_to_name: dict[int, str],
    category_colors: dict[int, tuple[int, int, int]],
):
    """
    绘制当前图片的全部 COCO 标注框和类别。

    返回：
        可视化图片
        实际绘制框数量
        当前图片类别统计
    """
    result = image.copy()
    image_height, image_width = result.shape[:2]

    drawn_count = 0
    category_counter = Counter()

    for annotation in annotations:
        if SKIP_CROWD and annotation.get("iscrowd", 0) == 1:
            continue

        if SKIP_IGNORE and annotation.get("ignore", 0) == 1:
            continue

        bbox = annotation.get("bbox")

        if bbox is None or len(bbox) != 4:
            continue

        try:
            bbox_area = float(bbox[2]) * float(bbox[3])
        except (TypeError, ValueError):
            continue

        if bbox_area < MIN_BBOX_AREA:
            continue

        clipped_bbox = clip_coco_bbox(
            bbox=bbox,
            image_width=image_width,
            image_height=image_height,
        )

        if clipped_bbox is None:
            continue

        x1, y1, x2, y2 = clipped_bbox

        category_id = annotation.get("category_id")

        category_name = category_id_to_name.get(
            category_id,
            f"unknown_{category_id}",
        )

        color = category_colors.get(
            category_id,
            (0, 255, 255),
        )

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            color,
            BOX_THICKNESS,
            cv2.LINE_AA,
        )

        label_parts = []

        if SHOW_CATEGORY_NAME:
            label_parts.append(category_name)

        if SHOW_ANNOTATION_ID:
            label_parts.append(
                f"ann:{annotation.get('id', 'unknown')}"
            )

        if SHOW_IMAGE_ID:
            label_parts.append(
                f"img:{annotation.get('image_id', 'unknown')}"
            )

        if label_parts:
            draw_label(
                image=result,
                text=" ".join(label_parts),
                x1=x1,
                y1=y1,
                color=color,
            )

        drawn_count += 1
        category_counter[category_name] += 1

    return result, drawn_count, category_counter


def draw_top_information(
    image,
    current_index: int,
    total_images: int,
    image_info: dict,
    object_count: int,
) -> None:
    """
    在图像左上角显示当前 split、文件名和目标数量。
    """
    if not SHOW_TOP_INFORMATION:
        return

    file_name = PurePosixPath(
        normalize_path(image_info.get("file_name", "unknown"))
    ).name

    image_id = image_info.get("id", "unknown")

    text = (
        f"{SPLIT} | "
        f"{current_index + 1}/{total_images} | "
        f"{file_name} | "
        f"image_id:{image_id} | "
        f"objects:{object_count}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    padding = 7

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    cv2.rectangle(
        image,
        (5, 5),
        (
            min(
                image.shape[1] - 1,
                5 + text_width + padding * 2,
            ),
            5 + text_height + baseline + padding * 2,
        ),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.putText(
        image,
        text,
        (
            5 + padding,
            5 + padding + text_height,
        ),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def resize_for_display(image):
    """
    大图缩小后显示，但不改变原始 bbox 坐标。
    """
    height, width = image.shape[:2]

    scale = min(
        MAX_SHOW_WIDTH / width,
        MAX_SHOW_HEIGHT / height,
        1.0,
    )

    if scale >= 1.0:
        return image

    return cv2.resize(
        image,
        dsize=None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# 7. 键盘控制
# ============================================================

def is_quit_key(key: int) -> bool:
    return key in {
        ord("q"),
        ord("Q"),
        27,
    }


def is_previous_key(key: int) -> bool:
    return key in {
        ord("a"),
        ord("A"),
        81,
        2424832,
    }


def is_next_key(key: int) -> bool:
    return key in {
        ord("d"),
        ord("D"),
        ord(" "),
        13,
        83,
        2555904,
    }


def is_jump_key(key: int) -> bool:
    return key in {
        ord("j"),
        ord("J"),
    }


# ============================================================
# 8. 数据集检查
# ============================================================

def filter_and_check_images(
    image_infos: list[dict],
    image_id_to_annotations,
    exact_index,
    basename_index,
    suffix_index,
):
    """
    根据当前 JSON 的 images 列表，查找对应实际图片。

    注意：
    这里不是遍历 IMAGE_ROOT 下所有图像，而是严格按照
    JSON 中的 images 顺序进行匹配。
    """
    resolved_items = []
    missing_items = []

    for image_info in image_infos:
        image_id = image_info.get("id")
        file_name = image_info.get("file_name")

        if not file_name:
            missing_items.append(
                (image_info, "file_name 为空")
            )
            continue

        annotations = image_id_to_annotations.get(
            image_id,
            [],
        )

        if (
            ONLY_SHOW_IMAGES_WITH_ANNOTATIONS
            and not annotations
        ):
            continue

        image_path = find_image_path(
            coco_file_name=file_name,
            exact_index=exact_index,
            basename_index=basename_index,
            suffix_index=suffix_index,
        )

        if image_path is None:
            missing_items.append(
                (image_info, "磁盘中未找到")
            )

            if SKIP_MISSING_IMAGES:
                continue

        resolved_items.append(
            {
                "image_info": image_info,
                "image_path": image_path,
            }
        )

    return resolved_items, missing_items


# ============================================================
# 9. 主程序
# ============================================================

def main() -> None:
    coco_data, annotation_source = load_selected_coco()

    validate_coco_data(coco_data)

    (
        image_infos,
        category_id_to_name,
        image_id_to_annotations,
    ) = build_coco_indexes(coco_data)

    category_colors = create_category_colors(
        category_id_to_name.keys(),
        seed=COLOR_SEED,
    )

    (
        exact_index,
        basename_index,
        suffix_index,
    ) = build_image_index(IMAGE_ROOT)

    resolved_items, missing_items = filter_and_check_images(
        image_infos=image_infos,
        image_id_to_annotations=image_id_to_annotations,
        exact_index=exact_index,
        basename_index=basename_index,
        suffix_index=suffix_index,
    )

    annotation_count = len(
        coco_data.get("annotations", [])
    )

    print("\n" + "=" * 75)
    print("SODA-D 数据集信息")
    print("=" * 75)
    print(f"当前划分：{SPLIT}")
    print(f"标注来源：{annotation_source}")
    print(f"JSON 中图片数量：{len(image_infos)}")
    print(f"JSON 中标注数量：{annotation_count}")
    print(f"类别数量：{len(category_id_to_name)}")
    print(f"成功匹配图片数量：{len(resolved_items)}")
    print(f"未找到图片数量：{len(missing_items)}")

    print("\n类别列表：")

    for category_id, category_name in sorted(
        category_id_to_name.items()
    ):
        print(f"  {category_id}: {category_name}")

    if missing_items:
        print("\n未找到图片示例：")

        for image_info, reason in missing_items[:20]:
            print(
                f"  image_id={image_info.get('id')}, "
                f"file_name={image_info.get('file_name')}, "
                f"原因={reason}"
            )

    if not resolved_items:
        raise RuntimeError(
            "没有找到可显示图片，请检查 IMAGE_ROOT 和 JSON 的 "
            "file_name 是否匹配。"
        )

    print("\n操作说明：")
    print("  d / 右方向键 / 空格 / 回车：下一张")
    print("  a / 左方向键：上一张")
    print("  j：跳转到指定序号")
    print("  q / Esc：退出")
    print("=" * 75)

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    current_index = 0
    total_images = len(resolved_items)

    while 0 <= current_index < total_images:
        item = resolved_items[current_index]

        image_info = item["image_info"]
        image_path = item["image_path"]

        if image_path is None:
            current_index += 1
            continue

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            print(f"[读取失败] {image_path}")
            current_index += 1
            continue

        image_id = image_info.get("id")

        annotations = image_id_to_annotations.get(
            image_id,
            [],
        )

        (
            visualized_image,
            object_count,
            category_counter,
        ) = draw_annotations(
            image=image,
            annotations=annotations,
            category_id_to_name=category_id_to_name,
            category_colors=category_colors,
        )

        draw_top_information(
            image=visualized_image,
            current_index=current_index,
            total_images=total_images,
            image_info=image_info,
            object_count=object_count,
        )

        display_image = resize_for_display(
            visualized_image
        )

        file_name = PurePosixPath(
            normalize_path(
                image_info.get("file_name", "unknown")
            )
        ).name

        window_title = (
            f"SODA-D {SPLIT} | "
            f"{current_index + 1}/{total_images} | "
            f"{file_name} | "
            f"objects:{object_count}"
        )

        try:
            cv2.setWindowTitle(
                WINDOW_NAME,
                window_title,
            )
        except cv2.error:
            pass

        cv2.imshow(
            WINDOW_NAME,
            display_image,
        )

        category_text = ", ".join(
            f"{category}:{count}"
            for category, count in sorted(
                category_counter.items()
            )
        )

        if not category_text:
            category_text = "无公开标注"

        print(
            f"[{current_index + 1}/{total_images}] "
            f"{image_info.get('file_name')} | "
            f"image_id={image_id} | "
            f"目标数={object_count} | "
            f"{category_text}"
        )

        key = cv2.waitKeyEx(0)

        if is_quit_key(key):
            break

        if is_previous_key(key):
            current_index = max(
                0,
                current_index - 1,
            )
            continue

        if is_jump_key(key):
            try:
                user_input = input(
                    f"请输入跳转序号 1～{total_images}："
                )

                target_index = int(user_input) - 1

                if 0 <= target_index < total_images:
                    current_index = target_index
                else:
                    print("输入序号超出范围。")

            except ValueError:
                print("请输入整数。")

            continue

        # 任意其他按键进入下一张
        current_index += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()