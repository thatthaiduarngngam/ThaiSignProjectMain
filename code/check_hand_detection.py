import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm  # สำหรับแสดงความคืบหน้า (ติดตั้งด้วย pip install tqdm)
import tensorflow as tf

# ตั้งค่า MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# โหลดโมเดลที่เทรนไว้
model = tf.keras.models.load_model('../sign_language_lstm_model.h5')

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

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # เตรียม VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # สถิติการตรวจจับ
    total_frames = 0
    hands_detected_frames = 0
    pose_detected_frames = 0
    problem_frames = []
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,  # ตรวจจับมือเดียว
        min_detection_confidence=0.5
    ) as hands, mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as pose:
        
        # ใช้ tqdm เพื่อแสดงความคืบหน้า
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=total, desc="Processing Video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                total_frames += 1
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # ตรวจจับมือและร่างกาย
                hands_results = hands.process(image_rgb)
                pose_results = pose.process(image_rgb)
                
                # วาด Landmark ของมือและร่างกาย
                if hands_results.multi_hand_landmarks and pose_results.pose_landmarks:
                    hands_detected_frames += 1
                    pose_detected_frames += 1
                    
                    # ดึง Landmark ของมือ
                    hand_landmarks = hands_results.multi_hand_landmarks[0]
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append([landmark.x, landmark.y, landmark.z])
                    
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
                    
                    # เตรียมข้อมูลสำหรับโมเดล
                    landmarks = np.expand_dims(relative_landmarks, axis=0)  # ขนาด (1, 21, 3)
                    
                    # ทำนาย
                    predictions = model.predict(landmarks)
                    predicted_label_idx = np.argmax(predictions[0])
                    predicted_label = label_map[predicted_label_idx]
                    
                    # แสดงผลการทำนาย
                    cv2.putText(frame, f"Prediction: {predicted_label}", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # วาด Landmark ของมือและร่างกาย
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0)),  # สีเขียวสำหรับมือ
                        mp_drawing.DrawingSpec(color=(255, 0, 0))  # สีน้ำเงินสำหรับจุด
                    )
                    mp_drawing.draw_landmarks(
                        frame,
                        pose_results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255)),  # สีแดงสำหรับร่างกาย
                        mp_drawing.DrawingSpec(color=(255, 0, 0))  # สีน้ำเงินสำหรับจุด
                    )
                else:
                    problem_frames.append(total_frames)
                    cv2.putText(frame, "Detection Issue", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                out.write(frame)
                pbar.update(1)
    
    cap.release()
    out.release()
    
    # สร้างรายงาน
    print(f"\n Detection Report:")
    print(f"- Total Frames: {total_frames}")
    print(f"- Hands Detected Frames: {hands_detected_frames} ({hands_detected_frames/total_frames:.1%})")
    print(f"- Pose Detected Frames: {pose_detected_frames} ({pose_detected_frames/total_frames:.1%})")
    print(f"- Problem Frames: {len(problem_frames)} (Frames: {problem_frames[:10]}{'...' if len(problem_frames)>10 else ''})")
    
    # เซฟเฟรมที่มีปัญหา
    if problem_frames:
        os.makedirs("problem_frames", exist_ok=True)
        cap = cv2.VideoCapture(input_path)
        for frame_num in problem_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(f"problem_frames/frame_{frame_num}.jpg", frame)
        cap.release()

# วิธีใช้งาน
input_video = "../rawvid/Bye.mp4"  # เปลี่ยน path ตามต้องการ Bye มีปัญหานะเพราะว่ามันจับไม่ถูก
output_video = "output_detection.mp4"
process_video(input_video, output_video)