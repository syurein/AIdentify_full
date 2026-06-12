import os
import glob
import torch
from PIL import Image
import numpy as np
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# ---------------------------------------------------------
# 設定項目
# ---------------------------------------------------------
# アノテーションしたい画像が格納されているフォルダ
IMAGE_DIR = "path/to/your/images" 

# 出力先のフォルダ（画像とYOLO用テキストがここに保存されます）
OUTPUT_DIR = "path/to/yolo_dataset"

# 検出したいオブジェクトのリスト（カンマ区切り、またはリスト形式）
TEXT_QUERIES = ["cat", "dog", "car"]

# 検出の閾値（これ以上のスコアのみアノテーションする）
SCORE_THRESHOLD = 0.15

# ---------------------------------------------------------
# 初期設定
# ---------------------------------------------------------
# クラス名とIDのマッピングを作成
CLASS_MAPPING = {name.strip(): idx for idx, name in enumerate(TEXT_QUERIES)}

# デバイスの設定
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# モデルとプロセッサの読み込み
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")

# 出力フォルダの作成
os.makedirs(OUTPUT_DIR, exist_ok=True)

# クラス定義ファイル（classes.txt）の書き出し
with open(os.path.join(OUTPUT_DIR, "classes.txt"), "w", encoding="utf-8") as f:
    for item in TEXT_QUERIES:
        f.write(f"{item.strip()}\n")

# 対応する画像拡張子の取得
image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"]
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

print(f"Found {len(image_paths)} images to process.")

# ---------------------------------------------------------
# メイン処理（推論 ＆ YOLOフォーマット変換）
# ---------------------------------------------------------
for img_path in image_paths:
    print(f"Processing: {os.path.basename(img_path)}...")
    
    # 1. 画像の読み込み（PIL経由でNumPy配列へ）
    try:
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)
    except Exception as e:
        print(f"Failed to read {img_path}: {e}")
        continue

    # 画像の元のサイズ（YOLOの正規化に必要）
    orig_h, orig_w = img_np.shape[:2]

    # 2. OWLv2用のターゲットサイズ設定と前処理
    size = max(orig_h, orig_w)
    target_sizes = torch.Tensor([[size, size]])
    inputs = processor(text=TEXT_QUERIES, images=img_np, return_tensors="pt").to(device)

    # 3. 推論実行
    with torch.no_grad():
        outputs = model(**inputs)

    # 後処理のためにCPUへ移動
    outputs.logits = outputs.logits.cpu()
    outputs.pred_boxes = outputs.pred_boxes.cpu()

    # 4. 結果の解析
    results = processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes)
    boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

    yolo_lines = []

    for box, score, label in zip(boxes, scores, labels):
        if score < SCORE_THRESHOLD:
            continue

        # 検出されたオブジェクトの名前と対応するクラスIDを取得
        class_name = TEXT_QUERIES[label.item()].strip()
        class_id = CLASS_MAPPING[class_name]

        # boxは [xmin, ymin, xmax, ymax] の絶対ピクセル座標
        xmin, ymin, xmax, ymax = box.tolist()

        # 5. YOLOフォーマット（中心X, 中心Y, 幅, 高さ）の正規化計算
        # ※OWLv2の出力が画像最大辺を基準にした正方形ベースになるケースを考慮し、元の画像サイズでクリップ
        xmin = max(0, min(xmin, orig_w))
        xmax = max(0, min(xmax, orig_w))
        ymin = max(0, min(ymin, orig_h))
        ymax = max(0, min(ymax, orig_h))

        box_w = xmax - xmin
        box_h = ymax - ymin
        x_center = xmin + (box_w / 2.0)
        y_center = ymin + (box_h / 2.0)

        # 0.0 〜 1.0 に正規化
        x_center_norm = x_center / orig_w
        y_center_norm = y_center / orig_h
        box_w_norm = box_w / orig_w
        box_h_norm = box_h / orig_h

        # YOLOフォーマットの行を生成
        yolo_lines.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {box_w_norm:.6f} {box_h_norm:.6f}")

    # 6. 結果の保存
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # 画像自体を出力フォルダへコピー（またはPILで保存）
    img_pil.save(os.path.join(OUTPUT_DIR, f"{base_name}.jpg"))

    # YOLO用テキストファイルの保存（空でも作成する仕様。背景画像として学習に使えるため）
    txt_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))

print(f"\nAnnotation finished! Dataset saved to: {OUTPUT_DIR}")