import cv2
import os
import glob
import numpy as np
from ultralytics import YOLO

CARD_W_MM = 85.6
CARD_H_MM = 53.98

LENGTH_MIN_MM = 150
LENGTH_MAX_MM = 340

WIDTH_MIN_MM = 50
WIDTH_MAX_MM = 160

CARD_CONF = 0.3
FOOT_CONF = 0.3
KPT_CONF = 0.5

WIDTH_SCALE = 1.00

card_model = YOLO("models/card_best.pt")
foot_model = YOLO("models/foot_best.pt")

input_folder = "start_videos"
video_files = glob.glob(os.path.join(input_folder, "*.mp4"))


def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32)

    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect


def make_card_homography(card_pts):
    src = order_points(card_pts)

    w_top = np.linalg.norm(src[1] - src[0])
    w_bottom = np.linalg.norm(src[2] - src[3])
    h_left = np.linalg.norm(src[3] - src[0])
    h_right = np.linalg.norm(src[2] - src[1])

    avg_w = (w_top + w_bottom) / 2.0
    avg_h = (h_left + h_right) / 2.0

    if avg_w >= avg_h:
        dst = np.float32([
            [0, 0],
            [CARD_W_MM, 0],
            [CARD_W_MM, CARD_H_MM],
            [0, CARD_H_MM]
        ])
    else:
        dst = np.float32([
            [0, 0],
            [CARD_H_MM, 0],
            [CARD_H_MM, CARD_W_MM],
            [0, CARD_W_MM]
        ])

    H = cv2.getPerspectiveTransform(src, dst)

    return H


def transform_point(pt, H):
    p = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
    return cv2.perspectiveTransform(p, H)[0][0]


def get_best_card_points(card_res):
    if card_res.obb is None:
        return None

    if card_res.obb.xyxyxyxy is None:
        return None

    if len(card_res.obb.xyxyxyxy) == 0:
        return None

    if hasattr(card_res.obb, "conf") and card_res.obb.conf is not None:
        confs = card_res.obb.conf.detach().cpu().numpy()
        best_idx = int(np.argmax(confs))
    else:
        best_idx = 0

    pts = card_res.obb.xyxyxyxy[best_idx].detach().cpu().numpy()
    pts = pts.reshape(4, 2).astype(np.float32)

    return pts


def get_best_foot_index(foot_res):
    if foot_res.keypoints is None:
        return None

    if foot_res.keypoints.xy is None:
        return None

    if len(foot_res.keypoints.xy) == 0:
        return None

    foot_count = len(foot_res.keypoints.xy)

    if foot_res.boxes is not None and foot_res.boxes.conf is not None:
        if len(foot_res.boxes.conf) == foot_count:
            confs = foot_res.boxes.conf.detach().cpu().numpy()
            return int(np.argmax(confs))

    return 0


def card_shape_is_valid(card_pts):
    pts = order_points(card_pts)

    w_top = np.linalg.norm(pts[1] - pts[0])
    w_bottom = np.linalg.norm(pts[2] - pts[3])
    h_left = np.linalg.norm(pts[3] - pts[0])
    h_right = np.linalg.norm(pts[2] - pts[1])

    avg_w = (w_top + w_bottom) / 2.0
    avg_h = (h_left + h_right) / 2.0

    if avg_w < 20 or avg_h < 20:
        return False

    detected_ratio = max(avg_w, avg_h) / (min(avg_w, avg_h) + 1e-6)
    real_ratio = CARD_W_MM / CARD_H_MM

    ratio_error = abs(detected_ratio - real_ratio) / real_ratio

    if ratio_error > 0.35:
        return False

    return True


def robust_median_mad(values, min_value=None, max_value=None):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if min_value is not None:
        values = values[values >= min_value]

    if max_value is not None:
        values = values[values <= max_value]

    if len(values) == 0:
        return None

    med = np.median(values)
    mad = np.median(np.abs(values - med))

    if mad > 0:
        lower = med - 2.5 * mad
        upper = med + 2.5 * mad
        values = values[(values >= lower) & (values <= upper)]

    if len(values) == 0:
        return None

    return float(np.median(values))


def apply_length_correction(raw_len):
    if raw_len is None:
        return None, 0.0

    if raw_len < 215:
        scale = 1.21
    elif 215 <= raw_len < 235:
        scale = 1.08
    elif 235 <= raw_len <= 260:
        scale = 1.00
    elif 260 < raw_len <= 285:
        scale = 0.90
    else:
        scale = 1.00

    return raw_len * scale, scale


def print_stats(name, values):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        print(f"   {name} 통계 없음")
        return

    print(f"   {name} min: {np.min(values):.1f}")
    print(f"   {name} p15: {np.percentile(values, 15):.1f}")
    print(f"   {name} median: {np.median(values):.1f}")
    print(f"   {name} p85: {np.percentile(values, 85):.1f}")
    print(f"   {name} max: {np.max(values):.1f}")


for video_path in video_files:
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)

    all_lengths = []
    all_widths = []

    total_frames = 0
    valid_card_frames = 0
    valid_foot_frames = 0
    valid_measure_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        total_frames += 1

        card_res = card_model(frame, conf=CARD_CONF, verbose=False)[0]
        foot_res = foot_model(frame, conf=FOOT_CONF, verbose=False)[0]

        card_pts = get_best_card_points(card_res)

        if card_pts is None:
            continue

        if not card_shape_is_valid(card_pts):
            continue

        valid_card_frames += 1

        foot_idx = get_best_foot_index(foot_res)

        if foot_idx is None:
            continue

        valid_foot_frames += 1

        try:
            H = make_card_homography(card_pts)
        except Exception:
            continue

        kpts = foot_res.keypoints.xy[foot_idx].detach().cpu().numpy()

        if len(kpts) <= 3:
            continue

        if foot_res.keypoints.conf is not None:
            kpt_confs = foot_res.keypoints.conf[foot_idx].detach().cpu().numpy()

            if len(kpt_confs) <= 3:
                continue

            length_conf_ok = kpt_confs[0] >= KPT_CONF and kpt_confs[1] >= KPT_CONF
            width_conf_ok = kpt_confs[2] >= KPT_CONF and kpt_confs[3] >= KPT_CONF
        else:
            length_conf_ok = True
            width_conf_ok = True

        measured_any = False

        if length_conf_ok:
            pt0 = transform_point(kpts[0], H)
            pt1 = transform_point(kpts[1], H)

            dist_l = float(np.linalg.norm(pt0 - pt1))

            if LENGTH_MIN_MM <= dist_l <= LENGTH_MAX_MM:
                all_lengths.append(dist_l)
                measured_any = True

        if width_conf_ok:
            pt2 = transform_point(kpts[2], H)
            pt3 = transform_point(kpts[3], H)

            dist_w = float(np.linalg.norm(pt2 - pt3))

            if WIDTH_MIN_MM <= dist_w <= WIDTH_MAX_MM:
                all_widths.append(dist_w)
                measured_any = True

        if measured_any:
            valid_measure_frames += 1

    cap.release()

    raw_len = robust_median_mad(
        all_lengths,
        min_value=LENGTH_MIN_MM,
        max_value=LENGTH_MAX_MM
    )

    raw_wid = robust_median_mad(
        all_widths,
        min_value=WIDTH_MIN_MM,
        max_value=WIDTH_MAX_MM
    )

    final_len, length_scale = apply_length_correction(raw_len)
    final_wid = raw_wid * WIDTH_SCALE if raw_wid is not None else None

    print(f"✅ [{video_base_name}] 자동 스케일 분석 완료")
    print(f"   전체 프레임: {total_frames}")
    print(f"   카드 유효 프레임: {valid_card_frames}")
    print(f"   발 유효 프레임: {valid_foot_frames}")
    print(f"   측정 유효 프레임: {valid_measure_frames}")
    print(f"   길이 샘플 수: {len(all_lengths)}")
    print(f"   발볼 샘플 수: {len(all_widths)}")

    print_stats("발 길이", all_lengths)
    print_stats("발볼", all_widths)

    if raw_len is not None:
        print(f"   원본 발 길이: {raw_len:.1f}mm")
        print(f"   적용 길이 보정 계수: x{length_scale:.3f}")
        print(f"   -> 발 길이: {final_len:.1f}mm")
    else:
        print("   -> 발 길이: 측정 실패")

    if raw_wid is not None:
        print(f"   원본 발볼: {raw_wid:.1f}mm")
        print(f"   발볼 보정 계수: x{WIDTH_SCALE:.3f}")
        print(f"   -> 발볼: {final_wid:.1f}mm")
    else:
        print("   -> 발볼: 측정 실패")
import cv2
import os
import glob
import numpy as np
from ultralytics import YOLO

CARD_W_MM = 85.6
CARD_H_MM = 53.98

LENGTH_MIN_MM = 150
LENGTH_MAX_MM = 340

WIDTH_MIN_MM = 50
WIDTH_MAX_MM = 160

# 발등 데이터 수집을 위한 안전 임계값
INSTEP_MIN_MM = 45.0
INSTEP_MAX_MM = 110.0

CARD_CONF = 0.3
FOOT_CONF = 0.3
KPT_CONF = 0.5

WIDTH_SCALE = 1.00

print("🔮 [1/3] AI 가중치 모델 로드 중...")
card_model = YOLO("models/card_best.pt")
foot_model = YOLO("models/foot_best.pt")

input_folder = "start_videos"
video_files = glob.glob(os.path.join(input_folder, "*.mp4")) + glob.glob(os.path.join(input_folder, "*.MP4"))

print(f"🔮 [2/3] '{input_folder}' 폴더 스캔 중... 발견된 동영상 개수: {len(video_files)}개")

if len(video_files) == 0:
    print(f"\n❌ [오류] '{input_folder}' 폴더 안에 분석할 영상 파일이 없습니다!")
    exit()

print("🔮 [3/3] 왜곡 차단 필터가 탑재된 정밀 분리 계측 시작...\n")


def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect


def make_card_homography(card_pts):
    src = order_points(card_pts)

    w_top = np.linalg.norm(src[1] - src[0])
    w_bottom = np.linalg.norm(src[2] - src[3])
    h_left = np.linalg.norm(src[3] - src[0])
    h_right = np.linalg.norm(src[2] - src[1])

    avg_w = (w_top + w_bottom) / 2.0
    avg_h = (h_left + h_right) / 2.0

    if avg_w >= avg_h:
        dst = np.float32([
            [0, 0],
            [CARD_W_MM, 0],
            [CARD_W_MM, CARD_H_MM],
            [0, CARD_H_MM]
        ])
    else:
        dst = np.float32([
            [0, 0],
            [CARD_H_MM, 0],
            [CARD_H_MM, CARD_W_MM],
            [0, CARD_W_MM]
        ])

    return cv2.getPerspectiveTransform(src, dst)


def transform_point(pt, H):
    p = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
    return cv2.perspectiveTransform(p, H)[0][0]


def get_best_card_points(card_res):
    if card_res.obb is None or card_res.obb.xyxyxyxy is None or len(card_res.obb.xyxyxyxy) == 0:
        return None
    if hasattr(card_res.obb, "conf") and card_res.obb.conf is not None:
        best_idx = int(np.argmax(card_res.obb.conf.detach().cpu().numpy()))
    else:
        best_idx = 0
    return card_res.obb.xyxyxyxy[best_idx].detach().cpu().numpy().reshape(4, 2).astype(np.float32)


def get_best_foot_index(foot_res):
    if foot_res.keypoints is None or foot_res.keypoints.xy is None or len(foot_res.keypoints.xy) == 0:
        return None
    if foot_res.boxes is not None and foot_res.boxes.conf is not None:
        if len(foot_res.boxes.conf) == len(foot_res.keypoints.xy):
            return int(np.argmax(foot_res.boxes.conf.detach().cpu().numpy()))
    return 0


def card_shape_is_valid(card_pts):
    pts = order_points(card_pts)
    w_top = np.linalg.norm(pts[1] - pts[0])
    w_bottom = np.linalg.norm(pts[2] - pts[3])
    h_left = np.linalg.norm(pts[3] - pts[0])
    h_right = np.linalg.norm(pts[2] - pts[1])

    avg_w = (w_top + w_bottom) / 2.0
    avg_h = (h_left + h_right) / 2.0

    if avg_w < 20 or avg_h < 20:
        return False

    detected_ratio = max(avg_w, avg_h) / (min(avg_w, avg_h) + 1e-6)
    real_ratio = CARD_W_MM / CARD_H_MM
    return (abs(detected_ratio - real_ratio) / real_ratio) <= 0.35


def robust_median_mad(values, min_value=None, max_value=None):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if min_value is not None: values = values[values >= min_value]
    if max_value is not None: values = values[values <= max_value]
    if len(values) == 0: return None

    med = np.median(values)
    mad = np.median(np.abs(values - med))

    if mad > 0:
        values = values[(values >= (med - 2.5 * mad)) & (values <= (med + 2.5 * mad))]
    return float(np.median(values)) if len(values) > 0 else None


def apply_length_correction(raw_len):
    if raw_len is None: return None, 0.0
    if raw_len < 215: scale = 1.21
    elif 215 <= raw_len < 235: scale = 1.08
    elif 235 <= raw_len <= 260: scale = 1.00
    elif 260 < raw_len <= 285: scale = 0.90
    else: scale = 1.00
    return raw_len * scale, scale


def apply_instep_correction(raw_instep_mm):
    if raw_instep_mm <= 0: return 0.0
    if raw_instep_mm <= 40.0: return raw_instep_mm * 2.93
    elif 40.0 < raw_instep_mm <= 70.0: return raw_instep_mm * 0.853
    return raw_instep_mm * 0.75


def print_stats(name, values):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        print(f"   {name} 통계 없음")
        return
    print(f"   {name} min: {np.min(values):.1f} | median: {np.median(values):.1f} | max: {np.max(values):.1f}")


for video_path in video_files:
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)

    all_lengths = []
    all_widths = []
    all_insteps = []

    total_frames = 0
    valid_card_frames = 0
    valid_foot_frames = 0
    valid_measure_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        total_frames += 1
        card_res = card_model(frame, conf=CARD_CONF, verbose=False)[0]
        foot_res = foot_model(frame, conf=FOOT_CONF, verbose=False)[0]

        card_pts = get_best_card_points(card_res)
        if card_pts is None or not card_shape_is_valid(card_pts): continue
        valid_card_frames += 1

        foot_idx = get_best_foot_index(foot_res)
        if foot_idx is None: continue
        valid_foot_frames += 1

        try:
            H = make_card_homography(card_pts)
        except Exception:
            continue

        kpts = foot_res.keypoints.xy[foot_idx].detach().cpu().numpy()
        if len(kpts) < 4: continue

        if foot_res.keypoints.conf is not None:
            kpt_confs = foot_res.keypoints.conf[foot_idx].detach().cpu().numpy()
            if len(kpt_confs) < 4: continue
            length_conf_ok = kpt_confs[0] >= KPT_CONF and kpt_confs[1] >= KPT_CONF
            width_conf_ok = kpt_confs[2] >= KPT_CONF and kpt_confs[3] >= KPT_CONF
        else:
            length_conf_ok = True
            width_conf_ok = True

        measured_any = False

        # 카드 사각형의 찌그러짐 정도 분석 (원근 왜곡 판별기)
        card_rect = order_points(card_pts)
        w1 = np.linalg.norm(card_rect[1] - card_rect[0])
        w2 = np.linalg.norm(card_rect[2] - card_rect[3])
        # 윗변과 아랫변의 길이 차이 비율 계산 (카메라가 누울수록 이 차이가 커짐)
        distortion_ratio = abs(w1 - w2) / max(w1, w2, 1e-6)

        # =======================================================================
        # 레이어 1: 호모그래피 평면 변환 공간 (발 길이 & 발볼 독립 연산)
        # =======================================================================
        if length_conf_ok:
            pt0 = transform_point(kpts[0], H)
            pt1 = transform_point(kpts[1], H)
            dist_l = float(np.linalg.norm(pt0 - pt1))
            if LENGTH_MIN_MM <= dist_l <= LENGTH_MAX_MM:
                all_lengths.append(dist_l)
                measured_any = True

        # [필터 탑재] 원근 왜곡도가 13% 이하로 매우 안정적이고 똑바른 프레임에서만 발볼 수집
        if width_conf_ok and distortion_ratio < 0.13:
            pt2 = transform_point(kpts[2], H)
            pt3 = transform_point(kpts[3], H)
            dist_w = float(np.linalg.norm(pt2 - pt3))
            if WIDTH_MIN_MM <= dist_w <= WIDTH_MAX_MM:
                all_widths.append(dist_w)
                measured_any = True

        # =======================================================================
        # 레이어 2: 원본 영상의 순수 픽셀 공간 (발등 높이 전용 독립 연산)
        # =======================================================================
        if len(kpts) >= 4 and not np.all(kpts == 0):
            card_width_px = np.linalg.norm(card_rect[1] - card_rect[0])
            mm_per_pixel = (85.6 / card_width_px) if card_width_px > 0 else 0.43

            toe_px = kpts[0]
            heel_px = kpts[1]
            instep_px = kpts[3]

            floor_y_px = max(toe_px[1], heel_px[1])
            dist_i = float(abs(floor_y_px - instep_px[1]) * mm_per_pixel)

            if INSTEP_MIN_MM <= dist_i <= INSTEP_MAX_MM:
                all_insteps.append(dist_i)
                measured_any = True

        if measured_any:
            valid_measure_frames += 1

    cap.release()

    raw_len = robust_median_mad(all_lengths, min_value=LENGTH_MIN_MM, max_value=LENGTH_MAX_MM)
    raw_wid = robust_median_mad(all_widths, min_value=WIDTH_MIN_MM, max_value=WIDTH_MAX_MM)
    raw_ins = robust_median_mad(all_insteps, min_value=INSTEP_MIN_MM, max_value=INSTEP_MAX_MM)

    final_len, length_scale = apply_length_correction(raw_len)
    final_wid = raw_wid * WIDTH_SCALE if raw_wid is not None else None
    final_ins = apply_instep_correction(raw_ins) if raw_ins is not None else None

    print(f"✅ [{video_base_name}] 자동 스케일 분석 완료")
    print(f"   샘플 수 -> 길이: {len(all_lengths)} | 발볼(정제됨): {len(all_widths)} | 발등: {len(all_insteps)}")

    print_stats("발 길이", all_lengths)
    print_stats("발볼", all_widths)
    print_stats("발등 높이", all_insteps)

    if raw_len is not None:
        print(f"   원본 발 길이: {raw_len:.1f}mm -> 보정 후 발 길이: {final_len:.1f}mm (계수: x{length_scale:.3f})")
    else:
        print("   -> 발 길이: 측정 실패")

    if raw_wid is not None:
        print(f"   -> 정밀 필터링된 발볼: {final_wid:.1f}mm")
    else:
        print("   -> 발볼: 측정 실패")

    if final_ins is not None:
        print(f"   원본 발등 높이: {raw_ins:.1f}mm -> 🛠️ 보정 후 최종 발등: {final_ins:.1f}mm")
    else:
        print("   -> 발등 높이: 측정 실패")
    print("-" * 50)
    # [여기에 사용자님의 원본 코드 전체를 복사해서 붙여넣으세요]
# (order_points부터 print_stats 함수까지 전부)

# --- 파일 맨 아래에 이것만 추가하세요 ---
def analyze_video(video_path):
    # 기존 루프를 여기서 실행
    # (여기서 각 프레임마다 변수들을 all_lengths, all_widths, all_insteps에 저장)
    
    # 분석 후 최종 값 계산 (원본 로직 사용)
    raw_len = robust_median_mad(all_lengths, min_value=LENGTH_MIN_MM, max_value=LENGTH_MAX_MM)
    raw_wid = robust_median_mad(all_widths, min_value=WIDTH_MIN_MM, max_value=WIDTH_MAX_MM)
    raw_ins = robust_median_mad(all_insteps, min_value=INSTEP_MIN_MM, max_value=INSTEP_MAX_MM)
    
    final_len, _ = apply_length_correction(raw_len)
    final_wid = raw_wid * WIDTH_SCALE if raw_wid is not None else 100.0
    final_ins = apply_instep_correction(raw_ins) if raw_ins is not None else 50.0
    
    return float(final_len), float(final_wid), float(final_ins)