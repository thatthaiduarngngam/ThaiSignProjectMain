import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import mediapipe as mp
import random

# ตั้งค่า MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# จุดอ้างอิงบนร่างกาย (ใช้จมูกเป็นจุดอ้างอิง)
BODY_REFERENCE_POINT = mp_pose.PoseLandmark.NOSE

# ฟังก์ชันหมุนภาพ
def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated_image = cv2.warpAffine(image, matrix, (cols, rows))
    return rotated_image

# ฟังก์ชันสะท้อนภาพ
def flip_image(image):
    return cv2.flip(image, 1)  # 1 คือสะท้อนในแนวนอน

# ฟังก์ชันเพิ่มข้อมูล (Data Augmentation)
def augment_data(image):
    if random.random() > 0.5:
        image = rotate_image(image, random.randint(-15, 15))  # หมุนภาพสุ่ม
    if random.random() > 0.5:
        image = flip_image(image)  # สะท้อนภาพ
    return image

# ฟังก์ชันดึง Landmark จากภาพ
def extract_landmarks(image_path, augment=False):
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # เพิ่มการทำ Data Augmentation (หมุนหรือสะท้อนภาพ)
    if augment:
        image = augment_data(image)
    
    # แปลงสีเป็น RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # ตรวจจับมือและร่างกาย
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands, \
        mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5) as pose:

        # ตรวจจับร่างกาย
        pose_results = pose.process(image_rgb)
        if not pose_results.pose_landmarks:
            return None

        # ดึงจุดอ้างอิงบนร่างกาย (จมูก)
        ref_point = pose_results.pose_landmarks.landmark[BODY_REFERENCE_POINT]
        ref_x, ref_y, ref_z = ref_point.x, ref_point.y, ref_point.z

        # ตรวจจับมือ
        hand_results = hands.process(image_rgb)
        if not hand_results.multi_hand_landmarks:
            return None

        # ดึง Landmark ของมือและคำนวณตำแหน่งสัมพันธ์กับร่างกาย
        landmarks = []
        for landmark in hand_results.multi_hand_landmarks[0].landmark:
            # คำนวณตำแหน่งสัมพันธ์กับจุดอ้างอิง
            rel_x = landmark.x - ref_x
            rel_y = landmark.y - ref_y
            rel_z = landmark.z - ref_z
            
            landmarks.append([rel_x, rel_y, rel_z])
        
        return np.array(landmarks)  # รูปร่าง (21, 3)

# โหลดข้อมูลและแปลงเป็น Landmark
def load_data(base_dir, augment=False):
    X = []
    y = []
    label_map = {}
    
    for label_idx, label_dir in enumerate(os.listdir(base_dir)):
        dir_path = os.path.join(base_dir, label_dir)
        
        if not os.path.isdir(dir_path):
            continue
        
        label_map[label_idx] = label_dir
        print(f"กำลังประมวลผลโฟลเดอร์: {label_dir}")
        
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            
            if not os.path.isfile(file_path):
                continue
            
            # ดึง Landmark พร้อมการทำ Data Augmentation
            landmarks = extract_landmarks(file_path, augment=augment)
            if landmarks is not None:
                X.append(landmarks)  # รูปร่าง (21, 3)
                y.append(label_idx)
    
    return np.array(X), np.array(y), label_map

# โหลดข้อมูล
X, y, label_map = load_data("../vidframes", augment=True)

# แปลง Label เป็น One-Hot Encoding
y = tf.keras.utils.to_categorical(y, num_classes=len(label_map))

# แบ่งข้อมูลเป็น Train/Validation
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล LSTM
model = Sequential([
    LSTM(64, input_shape=(21, 3)),  # 21 ขั้นเวลา, 3 features ต่อขั้นเวลา
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_map), activation='softmax')
])

# คอมไพล์โมเดล
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# เทรนโมเดล
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32
)

# บันทึกโมเดล
model.save("../sign_language_lstm_model.h5")