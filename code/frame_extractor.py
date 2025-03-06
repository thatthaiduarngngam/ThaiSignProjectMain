import os
import cv2

def extract_frames(video_path, output_dir, frame_interval=1):
    # สร้างโฟลเดอร์จากชื่อวิดีโอ (ไม่รวมนามสกุลไฟล์)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(output_dir, video_name)
    os.makedirs(save_dir, exist_ok=True)

    # อ่านวิดีโอด้วย OpenCV
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # เซฟเฟรมทุกๆ frame_interval เฟรม (default=1 คือทุกเฟรม)
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(save_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        
        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} frames from {video_name} to {save_dir}")

# ตั้งค่าการทำงานหลัก
input_dir = "rawvid"  # โฟลเดอร์เก็บวิดีโอต้นฉบับ
output_dir = "vidframes"    # โฟลเดอร์ปลายทางสำหรับเก็บเฟรม
video_extensions = ['.mp4', '.avi', '.mov']  # นามสกุลไฟล์วิดีโอที่รองรับ

# สร้างโฟลเดอร์ผลลัพธ์หากยังไม่มี
os.makedirs(output_dir, exist_ok=True)

# ประมวลผลทุกไฟล์ในโฟลเดอร์
for filename in os.listdir(input_dir):
    filepath = os.path.join(input_dir, filename)
    
    # ตรวจสอบนามสกุลไฟล์
    if os.path.isfile(filepath) and any(filename.lower().endswith(ext) for ext in video_extensions):
        extract_frames(filepath, output_dir, frame_interval=1)  # เปลี่ยน frame_interval เป็นค่าอื่นเพื่อลดจำนวนเฟรม
    else:
        print(f"Skipping non-video file: {filename}")