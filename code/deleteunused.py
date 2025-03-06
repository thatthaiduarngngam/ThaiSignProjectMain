import cv2
import os
import mediapipe as mp

# ตั้งค่า MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# โฟลเดอร์ที่เก็บเฟรมทั้งหมด (จากขั้นตอนก่อนหน้า)
base_dir = "../vidframes"

# ฟังก์ชันตรวจสอบมือในภาพ
def has_hands(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return False
    
    # แปลงสีเป็น RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # ตรวจจับมือ
    results = hands.process(image_rgb)
    
    return results.multi_hand_landmarks is not None

# วนลูปผ่านทุกโฟลเดอร์และไฟล์
for label_dir in os.listdir(base_dir):
    dir_path = os.path.join(base_dir, label_dir)
    
    if not os.path.isdir(dir_path):
        continue
    
    print(f"Processing directory: {label_dir}")
    deleted_count = 0
    
    # ตรวจสอบทุกไฟล์ในโฟลเดอร์
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        
        if not os.path.isfile(file_path):
            continue
        
        # ตรวจสอบว่ามีมือหรือไม่
        if not has_hands(file_path):
            os.remove(file_path)
            deleted_count += 1
            print(f"Deleted: {file_path}")
    
    print(f"Total deleted in {label_dir}: {deleted_count}\n")

hands.close()