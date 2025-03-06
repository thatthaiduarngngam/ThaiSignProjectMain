import os
import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2

# โหลดโมเดล
model = tf.keras.models.load_model('../sign_language_lstm_model.h5')

# ตั้งค่า MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ตั้งค่า MediaPipe Hands และ Pose
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5
)

# ตั้งค่า OpenCV เพื่อเปิด Webcam
cap = cv2.VideoCapture(0)

# ฟังก์ชันดึง Landmark จากภาพ
def extract_landmarks(image, mirror=False):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # ตรวจจับมือและร่างกาย
    hand_results = hands.process(image_rgb)
    pose_results = pose.process(image_rgb)
    
    if hand_results.multi_hand_landmarks and pose_results.pose_landmarks:
        # ดึง Landmark ของมือ
        hand_landmarks = hand_results.multi_hand_landmarks[0]
        landmarks = []
        for landmark in hand_landmarks.landmark:
            x, y, z = landmark.x, landmark.y, landmark.z
            if mirror:  # กลับมือ (mirror) ถ้าต้องการ
                x = 1 - x  # กลับ x-axis
            landmarks.append([x, y, z])
        
        # ดึงจุดอ้างอิงบนร่างกาย (จมูก)
        ref_point = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        ref_x, ref_y, ref_z = ref_point.x, ref_point.y, ref_point.z
        
        # คำนวณตำแหน่งมือสัมพันธ์กับร่างกาย
        relative_landmarks = []
        for landmark in landmarks:
            rel_x = landmark[0] - ref_x
            rel_y = landmark[1] - ref_y
            rel_z = landmark[2] - ref_z
            relative_landmarks.append([rel_x, rel_y, rel_z])
        
        return np.array(relative_landmarks), hand_landmarks, pose_results.pose_landmarks
    else:
        return None, None, None

# ฟังก์ชันแสดงผลลัพธ์
def draw_prediction(frame, label):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Prediction: {label}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

# สร้าง label_map จากชื่อโฟลเดอร์ใน vidframes
def create_label_map(base_dir):
    label_map = {}
    label_idx = 0
    for label_dir in sorted(os.listdir(base_dir)):  # เรียงลำดับตามชื่อโฟลเดอร์
        dir_path = os.path.join(base_dir, label_dir)
        if os.path.isdir(dir_path):
            label_map[label_idx] = label_dir
            label_idx += 1
    return label_map

# โหลด label_map จาก vidframes
label_map = create_label_map('../vidframes')

# Main loop สำหรับ Webcam
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # ดึง Landmark โดยการกลับมือ (mirror) ถ้าจำเป็น
    landmarks, hand_landmarks, pose_landmarks = extract_landmarks(frame, mirror=True)  # ใช้ mirror=True เมื่อสลับมือ
    
    if landmarks is not None:
        # เตรียมข้อมูลสำหรับโมเดล
        landmarks = np.expand_dims(landmarks, axis=0)  # ขนาด (1, 21, 3)
        
        # ทำนาย
        predictions = model.predict(landmarks)
        
        # ดึงผลลัพธ์ที่ทำนายได้ (สมมติว่าเป็นการทำนายที่ดีที่สุด)
        predicted_label_idx = np.argmax(predictions[0])
        
        # ใช้ label_map ที่สร้างจากชื่อโฟลเดอร์ใน vidframes
        predicted_label = label_map[predicted_label_idx]
        
        # แสดงผลลัพธ์
        draw_prediction(frame, predicted_label)
        
        # วาด Landmark ของมือและร่างกายลงในภาพ
        if hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0, 255, 0)),  # สีเขียวสำหรับมือ
                                    mp_drawing.DrawingSpec(color=(0, 255, 0)))
        
        if pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(255, 0, 0)),  # สีน้ำเงินสำหรับร่างกาย
                                    mp_drawing.DrawingSpec(color=(255, 0, 0)))

    # แสดงผลภาพจาก Webcam
    cv2.imshow('Webcam Hand Gesture Prediction', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิด Webcam
cap.release()
cv2.destroyAllWindows()