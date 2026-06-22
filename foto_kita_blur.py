import cv2
import mediapipe as mp
import pygame
import os
import time
import sys

# ================= KONFIGURASI =================
SONG_PATH = "foto_kita_blur.mp3"
MODEL_PATH = "hand_landmarker.task"
BLUR_STRENGTH = (55, 55)
COOLDOWN = 2.0
DETECT_EVERY_N_FRAMES = 2  # Deteksi setiap 2 frame (hemat CPU)

# ================= CEK MODEL =================
if not os.path.exists(MODEL_PATH):
    print("❌ Model tidak ditemukan! Jalankan: uv run download_model.py")
    sys.exit(1)

# ================= AUDIO =================
pygame.mixer.init()
if os.path.exists(SONG_PATH):
    pygame.mixer.music.load(SONG_PATH)
    pygame.mixer.music.play(-1)
    print("🎵 Lagu diputar...")

# ================= MEDIAPIPE =================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5
)

# ================= KAMERA (RESOLUSI LEBIH KECIL = LEBIH CEPAT) =================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Kamera gagal!")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Turunkan resolusi
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

last_capture_time = 0
frame_timestamp_ms = 0
frame_count = 0
photo_saved_for_this_gesture = False
current_blur_status = False  # Status blur terakhir

print("\n📸 MODE LIVE - Layar Bersih")
print("✌️  Pose 2 jari → Blur real-time")
print("🔴 Tekan 'q' untuk keluar\n")

try:
    with HandLandmarker.create_from_options(options) as hand_landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            frame_count += 1

            # Deteksi tangan hanya setiap N frame (hemat CPU)
            if frame_count % DETECT_EVERY_N_FRAMES == 0:
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, 
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
                frame_timestamp_ms += 33 * DETECT_EVERY_N_FRAMES
                result = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                gesture_detected = False
                if result.hand_landmarks:
                    for hand_landmarks in result.hand_landmarks:
                        # Deteksi pose 2 jari (tanpa gambar kerangka)
                        lm = hand_landmarks
                        index_up = lm[8].y < lm[6].y
                        mid_up = lm[12].y < lm[10].y
                        ring_down = lm[16].y > lm[14].y
                        pinky_down = lm[20].y > lm[18].y
                        gesture_detected = index_up and mid_up and ring_down and pinky_down

                current_blur_status = gesture_detected

            # ================= TERAPKAN BLUR =================
            if current_blur_status:
                frame = cv2.GaussianBlur(frame, BLUR_STRENGTH, 0)
                
                now = time.time()
                if not photo_saved_for_this_gesture and (now - last_capture_time > COOLDOWN):
                    last_capture_time = now
                    photo_saved_for_this_gesture = True
                    filename = f"foto_blur_{int(now)}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"✅ Foto tersimpan: {filename}")
            else:
                photo_saved_for_this_gesture = False

            # ================= INDIKATOR KECIL DI POJOK =================
            # Hanya titik kecil + teks mini, tidak mengganggu view
            indicator_x, indicator_y = w - 100, 30
            color = (0, 0, 255) if current_blur_status else (0, 255, 0)
            cv2.circle(frame, (indicator_x, indicator_y), 8, color, -1)
            cv2.putText(frame, "BLUR" if current_blur_status else "OK", 
                        (indicator_x + 15, indicator_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            cv2.imshow("Foto Kita Blur", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\n⚠️  Program dihentikan")

finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.music.stop()
    print("👋 Selesai!")