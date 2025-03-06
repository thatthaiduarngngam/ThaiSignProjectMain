import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. ตั้งค่า Path
input_folder = "../filtered_frames"  # โฟลเดอร์ที่เก็บเฟรมที่กรองแล้ว
output_model_path = "../skeleton_lstm_model.h5"  # Path สำหรับบันทึกโมเดล

# 2. กำหนดค่าคงที่
NUM_LANDMARKS = 21  # MediaPipe Hands มี 21 Landmarks ต่อมือ
NUM_DIMENSIONS = 3  # (x, y, z)
NUM_HANDS = 2  # จำนวนมือที่ต้องการ
NUM_FEATURES = NUM_LANDMARKS * NUM_DIMENSIONS * NUM_HANDS  # จำนวน Features ทั้งหมด

# 3. ฟังก์ชันสกัด Skeleton ด้วย MediaPipe
mp_hands = mp.solutions.hands

def extract_skeleton(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        skeleton = []
        
        # ตรวจสอบมือที่พบ
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # แยกค่า x, y, z ของแต่ละ Landmark เป็นตัวเลขเดี่ยว
                for lm in hand_landmarks.landmark:
                    skeleton.append(lm.x)
                    skeleton.append(lm.y)
                    skeleton.append(lm.z)
        
        # เติมค่า 0 หากไม่พบมือครบ 2 ข้าง (21 Landmarks × 3 มิติ × 2 มือ = 126 Features)
        expected_length = 2 * 21 * 3  # 126 Features
        while len(skeleton) < expected_length:
            skeleton.append(0.0)
        
        return np.array(skeleton, dtype=np.float32)

# 4. โหลดข้อมูลและเตรียม Dataset
def load_dataset(input_folder):
    sequences = []
    labels = []
    label_names = []
    
    for label_idx, folder_name in enumerate(sorted(os.listdir(input_folder))):
        folder_path = os.path.join(input_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        sequence = []
        for img_name in sorted(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            skeleton = extract_skeleton(img)
            sequence.append(skeleton)
            
        sequences.append(sequence)
        labels.append(label_idx)
        label_names.append(folder_name)
    
    return sequences, np.array(labels), label_names

# 5. โหลดข้อมูล
sequences, labels, label_names = load_dataset(input_folder)
print(f"Loaded {len(sequences)} sequences with {len(label_names)} classes: {label_names}")

# 6. ตรวจสอบจำนวน Features ในแต่ละ Sequence
for seq in sequences:
    for frame in seq:
        if len(frame) != NUM_FEATURES:
            raise ValueError(f"Invalid number of features: {len(frame)} (expected {NUM_FEATURES})")

# 7. Padding ให้ทุก Sequence ยาวเท่ากัน
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_seq_length = max(len(seq) for seq in sequences)
X_padded = pad_sequences(sequences, maxlen=max_seq_length, padding='post', dtype='float32')

# 8. แบ่งข้อมูลเป็น Train/Test
from sklearn.model_selection import train_test_split
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(X_padded, labels_encoded, test_size=0.2, random_state=42)

# 9. สร้างและเทรนโมเดล
input_shape = (max_seq_length, NUM_FEATURES)
num_classes = len(label_names)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=input_shape),
    LSTM(32),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 10. เทรนโมเดล
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)]
)



############ Save Model #################
# 7. บันทึกโมเดล
model.save(output_model_path)
print(f"Model saved to {output_model_path}")



from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder

# โหลดโมเดล
model = load_model("../skeleton_lstm_model.h5")

# โหลดข้อมูล Label Names
label_names = ["Bye", "Hungry"]  # ตัวอย่าง Label Names
le = LabelEncoder()
le.fit(label_names)  # Fit LabelEncoder ด้วย Label Names

# ข้อมูลทดสอบ (X_test ควรเป็นข้อมูลที่เตรียมไว้แล้ว)
predictions = model.predict(X_test)

# แปลงผลลัพธ์เป็น Class Index
predicted_indices = np.argmax(predictions, axis=1)

# แปลง Class Index กลับเป็น Label
predicted_labels = le.inverse_transform(predicted_indices)

# แสดงผลลัพธ์
for i, (true_label, pred_label) in enumerate(zip(y_test, predicted_labels)):
    print(f"ตัวอย่างที่ {i+1}: คาดการณ์ = {pred_label}, จริง = {le.inverse_transform([true_label])[0]}")

import pickle

# บันทึก Label Names
with open("../label_names.pkl", "wb") as f:
    pickle.dump(label_names, f)

# โหลด Label Names
with open("../label_names.pkl", "rb") as f:
    label_names = pickle.load(f)

import pandas as pd

results = pd.DataFrame({
    "True Label": le.inverse_transform(y_test),
    "Predicted Label": predicted_labels
})
print(results)