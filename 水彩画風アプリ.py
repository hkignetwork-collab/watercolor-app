import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Photoshop再現 水彩画ジェネレーター", layout="centered")
st.title("🎨 水彩画風フォト変換アプリ")
st.write("不自然な黒い線を排除し、色の境界に極細の陰影をつけることで表情をはっきりと残します。")

uploaded_file = st.file_uploader("写真をアップロードしてください", type=["jpg", "jpeg", "png"])

def generate_watercolor_paper_texture(height, width):
    """水彩紙の自然な凹凸を生成"""
    low_h, low_w = max(1, height // 2), max(1, width // 2)
    np.random.seed(42)
    # 大きな紙のうねり
    base = np.random.normal(128, 30, (low_h, low_w)).astype(np.float32)
    base = cv2.GaussianBlur(base, (3, 3), 0)
    paper = cv2.resize(base, (width, height), interpolation=cv2.INTER_CUBIC)

    # 表面の細かなざらつき
    fine = np.random.normal(0, 10, (height, width)).astype(np.float32)
    paper = paper + fine

    # 乗算用に調整（コントラストを少し抑えめにし、白浮きを防ぐ）
    paper_light = (paper - 128.0) / 128.0 * 0.15 + 1.0
    return np.clip(paper_light, 0.7, 1.3)

def apply_photoshop_watercolor_exact(img_bytes):
    file_bytes = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    h, w = img.shape[:2]
    max_dim = 1200
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # 1. 色の鮮やかさを少しだけ上げる
    img_contrast = cv2.convertScaleAbs(img, alpha=1.1, beta=0)

    # 2. にじみと平滑化（顔のディテールを壊さないよう、フィルタの範囲を少し小さく設定）
    paint = cv2.bilateralFilter(img_contrast, d=5, sigmaColor=50, sigmaSpace=50)
    paint = cv2.bilateralFilter(paint, d=5, sigmaColor=50, sigmaSpace=50)
    paint = cv2.medianBlur(paint, 3)

    # 3. 【黒線の排除と極細陰影の作成】
    gray = cv2.cvtColor(img_contrast, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # ラプラシアンフィルタで極細の「色の境目」だけを抽出
    edges = cv2.Laplacian(gray_blur, cv2.CV_16S, ksize=3)
    edges = cv2.convertScaleAbs(edges)
    
    # 境目の部分を真っ黒にするのではなく、「元の色を最大でも25%ほど暗くする」ための係数を作成
    edges_float = edges.astype(np.float32) / 255.0
    darken_factor = 1.0 - (edges_float * 0.25) 

    # 4. 【合成処理】
    paint_float = paint.astype(np.float32)
    for c in range(3):
        # 色の境目にのみ極細の陰影を落とす
        paint_float[:, :, c] = paint_float[:, :, c] * darken_factor
    
    # 水彩紙テクスチャを乗算で重ねる
    paper = generate_watercolor_paper_texture(h, w)
    for c in range(3):
        paint_float[:, :, c] = np.clip(paint_float[:, :, c] * paper, 0, 255)

    final_art = cv2.convertScaleAbs(paint_float, alpha=1.02, beta=2)

    # Web表示用 (RGB) とダウンロード用バイナリ
    rgb_img = cv2.cvtColor(final_art, cv2.COLOR_BGR2RGB)
    _, encoded_img = cv2.imencode('.jpg', final_art, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return rgb_img, encoded_img.tobytes()

if uploaded_file is not None:
    with st.spinner("Photoshop風の水彩画を作成中..."):
        result_img, download_bytes = apply_photoshop_watercolor_exact(uploaded_file.read())

    st.subheader("完成イメージ")
    st.image(result_img, use_container_width=True)

    st.download_button(
        label="水彩画を保存する",
        data=download_bytes,
        file_name="photoshop_watercolor_art.jpg",
        mime="image/jpeg"
    )
