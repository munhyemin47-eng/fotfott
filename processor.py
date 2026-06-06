import cv2
import numpy as np
from ultralytics import YOLO

# 1. 모델 로드 (전역에서 1회 로드)
card_model = YOLO("models/card_best.pt")
foot_model = YOLO("models/foot_best.pt")

# 2. 상수 정의
CARD_W_MM, CARD_H_MM = 85.6, 53.98
LENGTH_MIN_MM, LENGTH_MAX_MM = 150, 340
WIDTH_MIN_MM, WIDTH_MAX_MM = 50, 160
INSTEP_MIN_MM, INSTEP_MAX_MM = 45.0, 110.0
CARD_CONF, FOOT_CONF, KPT_CONF = 0.3, 0.3, 0.5
WIDTH_SCALE = 1.00

# 3. 모든 보조 함수들 (누락 없이 전부 포함)
def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1); d = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]; rect[3] = pts[np.argmax(d)]
    return rect

def make_card_homography(card_pts):
    src = order_points(card_pts)
    dst = np.float32([[0, 0], [CARD_W_MM, 0], [CARD_W_MM, CARD_H_MM], [0, CARD_H_MM]])
    return cv2.getPerspectiveTransform(src, dst)

def transform_point(pt, H):
    p = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
    return cv2.perspectiveTransform(p, H)[0][0]

def get_best_card_points(card_res):
    if card_res.obb is None or card_res.obb.xyxyxyxy is None or len(card_res.obb.xyxyxyxy) == 0: return None
    best_idx = int(np.argmax(card_res.obb.conf.detach().cpu().numpy())) if hasattr(card_res.obb, "conf") and card_res.obb.conf is not None else 0
    return card_res.obb.xyxyxyxy[best_idx].detach().cpu().numpy().reshape(4, 2).astype(np.float32)

def get_best_foot_index(foot_res):
    if foot_res.keypoints is None or foot_res.keypoints.xy is None or len(foot_res.keypoints.xy) == 0: return None
    if foot_res.boxes is not None and foot_res.boxes.conf is not None and len(foot_res.boxes.conf) == len(foot_res.keypoints.xy):
        return int(np.argmax(foot_res.boxes.conf.detach().cpu().numpy()))
    return 0

def card_shape_is_valid(card_pts):
    pts = order_points(card_pts)
    w_top, w_bottom = np.linalg.norm(pts[1]-pts[0]), np.linalg.norm(pts[2]-pts[3])
    h_left, h_right = np.linalg.norm(pts[3]-pts[0]), np.linalg.norm(pts[2]-pts[1])
    avg_w, avg_h = (w_top + w_bottom)/2.0, (h_left + h_right)/2.0
    if avg_w < 20 or avg_h < 20: return False
    ratio_error = abs((max(avg_w, avg_h) / (min(avg_w, avg_h) + 1e-6)) - (CARD_W_MM/CARD_H_MM)) / (CARD_W_MM/CARD_H_MM)
    return ratio_error <= 0.35

def robust_median_mad(values, min_value=None, max_value=None):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if min_value is not None: values = values[values >= min_value]
    if max_value is not None: values = values[values <= max_value]
    if len(values) == 0: return None
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad > 0: values = values[(values >= (med - 2.5 * mad)) & (values <= (med + 2.5 * mad))]
    return float(np.median(values)) if len(values) > 0 else None

def apply_length_correction(raw_len):
    if raw_len is None: return None, 0.0
    if raw_len < 215: scale = 1.21
    elif raw_len < 235: scale = 1.08
    elif raw_len <= 260: scale = 1.00
    elif raw_len <= 285: scale = 0.90
    else: scale = 1.00
    return raw_len * scale, scale

def apply_instep_correction(raw_instep_mm):
    if raw_instep_mm <= 0: return 0.0
    if raw_instep_mm <= 40.0: return raw_instep_mm * 2.93
    elif raw_instep_mm <= 70.0: return raw_instep_mm * 0.853
    return raw_instep_mm * 0.75

# 4. 분석 메인 함수 (이제 모든 보조 함수가 위에 정의되어 있어 에러가 나지 않습니다)
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    all_lengths, all_widths, all_insteps = [], [], []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        card_res = card_model(frame, conf=CARD_CONF, verbose=False)[0]
        foot_res = foot_model(frame, conf=FOOT_CONF, verbose=False)[0]
        
        card_pts = get_best_card_points(card_res)
        if card_pts is None or not card_shape_is_valid(card_pts): continue
        
        foot_idx = get_best_foot_index(foot_res)
        if foot_idx is None: continue
        
        try: H = make_card_homography(card_pts)
        except Exception: continue
        
        kpts = foot_res.keypoints.xy[foot_idx].detach().cpu().numpy()
        if len(kpts) < 4: continue
        
        # 길이 측정
        pt0, pt1 = transform_point(kpts[0], H), transform_point(kpts[1], H)
        dist_l = float(np.linalg.norm(pt0 - pt1))
        if LENGTH_MIN_MM <= dist_l <= LENGTH_MAX_MM: all_lengths.append(dist_l)
        
        # 발볼 측정
        pt2, pt3 = transform_point(kpts[2], H), transform_point(kpts[3], H)
        dist_w = float(np.linalg.norm(pt2 - pt3))
        if WIDTH_MIN_MM <= dist_w <= WIDTH_MAX_MM: all_widths.append(dist_w)
        
        # 발등 높이 측정
        card_rect = order_points(card_pts)
        mm_per_pixel = 85.6 / np.linalg.norm(card_rect[1] - card_rect[0])
        dist_i = abs(max(kpts[0][1], kpts[1][1]) - kpts[3][1]) * mm_per_pixel
        if INSTEP_MIN_MM <= dist_i <= INSTEP_MAX_MM: all_insteps.append(float(dist_i))
            
    cap.release()
    
    final_len, _ = apply_length_correction(robust_median_mad(all_lengths, LENGTH_MIN_MM, LENGTH_MAX_MM))
    final_wid = (robust_median_mad(all_widths, WIDTH_MIN_MM, WIDTH_MAX_MM) * WIDTH_SCALE) if robust_median_mad(all_widths, WIDTH_MIN_MM, WIDTH_MAX_MM) else 0.0
    final_ins = apply_instep_correction(robust_median_mad(all_insteps, INSTEP_MIN_MM, INSTEP_MAX_MM))
    
    return float(final_len or 0.0), float(final_wid or 0.0), float(final_ins or 0.0)