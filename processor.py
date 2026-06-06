import cv2
import numpy as np
from ultralytics import YOLO

# 모델 로드 (전역에서 1회만 로드)
card_model = YOLO("models/card_best.pt")
foot_model = YOLO("models/foot_best.pt")

# 상수값 정의
CARD_W_MM, CARD_H_MM = 85.6, 53.98
LENGTH_MIN_MM, LENGTH_MAX_MM = 150, 340
WIDTH_MIN_MM, WIDTH_MAX_MM = 50, 160
INSTEP_MIN_MM, INSTEP_MAX_MM = 45.0, 110.0
CARD_CONF, FOOT_CONF, KPT_CONF = 0.3, 0.3, 0.5
WIDTH_SCALE = 1.00

# --- [이하 기존 보조 함수들 (order_points, make_card_homography, 등등 그대로 유지)] ---
# (사용자님이 작성하신 보조 함수들이 이 사이에 있어야 합니다)

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
        
        # 1. 길이 계산 (레이어 1)
        pt0, pt1 = transform_point(kpts[0], H), transform_point(kpts[1], H)
        dist_l = float(np.linalg.norm(pt0 - pt1))
        if LENGTH_MIN_MM <= dist_l <= LENGTH_MAX_MM: all_lengths.append(dist_l)
        
        # 2. 발볼 계산 (원근 왜곡 필터 탑재)
        card_rect = order_points(card_pts)
        w1 = np.linalg.norm(card_rect[1] - card_rect[0])
        w2 = np.linalg.norm(card_rect[2] - card_rect[3])
        distortion_ratio = abs(w1 - w2) / max(w1, w2, 1e-6)
        
        if distortion_ratio < 0.13:
            pt2, pt3 = transform_point(kpts[2], H), transform_point(kpts[3], H)
            dist_w = float(np.linalg.norm(pt2 - pt3))
            if WIDTH_MIN_MM <= dist_w <= WIDTH_MAX_MM: all_widths.append(dist_w)
            
        # 3. 발등 높이 계산 (레이어 2 - 순수 픽셀 공간)
        mm_per_pixel = 85.6 / (w1 if w1 > 0 else 1.0)
        dist_i = float(abs(max(kpts[0][1], kpts[1][1]) - kpts[3][1]) * mm_per_pixel)
        if INSTEP_MIN_MM <= dist_i <= INSTEP_MAX_MM: all_insteps.append(dist_i)

    cap.release()
    
    # 최종 보정 및 결과 반환
    raw_len = robust_median_mad(all_lengths, min_value=LENGTH_MIN_MM, max_value=LENGTH_MAX_MM)
    raw_wid = robust_median_mad(all_widths, min_value=WIDTH_MIN_MM, max_value=WIDTH_MAX_MM)
    raw_ins = robust_median_mad(all_insteps, min_value=INSTEP_MIN_MM, max_value=INSTEP_MAX_MM)
    
    final_len, _ = apply_length_correction(raw_len)
    final_wid = (raw_wid * WIDTH_SCALE) if raw_wid is not None else 0.0
    final_ins = apply_instep_correction(raw_ins) if raw_ins is not None else 0.0
    
    return float(final_len or 0.0), float(final_wid or 0.0), float(final_ins or 0.0)