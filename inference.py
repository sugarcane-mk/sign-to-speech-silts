import os
import json
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json

label_map = {
    "0": "வழி",
    "1": "வழியா",
    "2": "மதியம்",
    "3": "அதிகமா",
    "4": "கை",
    "5": "பெருசா",
    "6": "பில்",
    "7": "ரசீது",
    "8": "சரி",
    "9": "செலவு",
    "10": "மாலை",
    "11": "ஃபேன்",
    "12": "எப்படி இருக்கு",
    "13": "நல்லா",
    "14": "சூப்பர்",
    "15": "மகிழ்ச்சி",
    "16": "வணக்கம்",
    "17": "வீடு",
    "18": "தெரியல",
    "19": "அதிகம்",
    "20": "மிகவும்",
    "21": "வேணும்",
    "22": "கம்மி வெயிட்",
    "23": "வெளிச்சம்",
    "24": "காலைல",
    "25": "நைட்",
    "26": "ஆன் பண்ணு",
    "27": "ஓடு",
    "28": "தூங்க",
    "29": "நேரா",
    "30": "ஸ்விட்சு",
    "31": "நன்றி",
    "32": "ரொம்ப நன்றி",
    "33": "சோகம்",
    "34": "சுடுது",
    "35": "அதிகமா சுடுது",
    "36": "வெயிட் பண்ணு",
    "37": "கொஞ்சம் வெயிட் பண்ணு",
    "38": "நட",
    "39": "தண்ணி",
    "40": "எப்போ",
    "41": "எழுது",
    "42": "ஆமா",
    "43": "ஓகே"
}

with open("label_map_tamil.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)


# print("label_map_tamil.json created successfully with Tamil slang terms!")
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

mp_holistic = mp.solutions.holistic
POSE_LANDMARKS_LEFT = {11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
POSE_LANDMARKS_RIGHT = {12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}
LEFT_COLOR = (0, 255, 0)
RIGHT_COLOR = (0, 0, 255)
MIDDLE_COLOR = (255, 255, 255)

mobilenet = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

def to_pixel_coords(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)

def draw_custom_skeleton(pose_landmarks, image_w, image_h, frame):
    def midpoint(lm1, lm2):
        return [(lm1.x + lm2.x) / 2, (lm1.y + lm2.y) / 2]
    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]
    neck_px = to_pixel_coords(type('LM', (object,), {'x': (left_shoulder.x + right_shoulder.x) / 2, 'y': (left_shoulder.y + right_shoulder.y) / 2})(), image_w, image_h)

    left_hip = pose_landmarks[23]
    right_hip = pose_landmarks[24]
    pelvis_px = to_pixel_coords(type('LM', (object,), {'x': (left_hip.x + right_hip.x) / 2, 'y': (left_hip.y + right_hip.y) / 2})(), image_w, image_h)

    nose = pose_landmarks[mp_holistic.PoseLandmark.NOSE]
    nose_px = to_pixel_coords(nose, image_w, image_h)

    cv2.circle(frame, neck_px, 6, MIDDLE_COLOR, -1)
    cv2.circle(frame, pelvis_px, 6, MIDDLE_COLOR, -1)
    cv2.line(frame, neck_px, nose_px, MIDDLE_COLOR, 3)
    cv2.line(frame, neck_px, to_pixel_coords(left_shoulder, image_w, image_h), MIDDLE_COLOR, 3)
    cv2.line(frame, neck_px, to_pixel_coords(right_shoulder, image_w, image_h), MIDDLE_COLOR, 3)
    cv2.line(frame, pelvis_px, to_pixel_coords(left_hip, image_w, image_h), MIDDLE_COLOR, 3)
    cv2.line(frame, pelvis_px, to_pixel_coords(right_hip, image_w, image_h), MIDDLE_COLOR, 3)
    cv2.line(frame, pelvis_px, neck_px, MIDDLE_COLOR, 3)

def draw_hand(hand_landmarks, image_w, image_h, frame, color):
    if hand_landmarks:
        for lm in hand_landmarks.landmark:
            x, y = to_pixel_coords(lm, image_w, image_h)
            cv2.circle(frame, (x, y), 4, color, -1)
        for connection in mp_holistic.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_lm = hand_landmarks.landmark[start_idx]
            end_lm = hand_landmarks.landmark[end_idx]
            start_xy = to_pixel_coords(start_lm, image_w, image_h)
            end_xy = to_pixel_coords(end_lm, image_w, image_h)
            cv2.line(frame, start_xy, end_xy, color, 2)

def read_video_frames(video_path, max_duration_sec=7):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    max_frames = int(fps * max_duration_sec)
    frames, count = [], 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1
    cap.release()
    return frames

def extract_mediapipe_keypoints(frames):
    processed_frames = []
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        for frame in frames:
            image_h, image_w = frame.shape[:2]
            black_frame = np.zeros_like(frame)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True

            if results.pose_landmarks:
                pose_landmarks = results.pose_landmarks.landmark
                for idx, lm in enumerate(pose_landmarks):
                    x, y = to_pixel_coords(lm, image_w, image_h)
                    color = LEFT_COLOR if idx in POSE_LANDMARKS_LEFT else RIGHT_COLOR if idx in POSE_LANDMARKS_RIGHT else MIDDLE_COLOR
                    cv2.circle(black_frame, (x, y), 5, color, -1)
                for connection in mp_holistic.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    start_xy = to_pixel_coords(pose_landmarks[start_idx], image_w, image_h)
                    end_xy = to_pixel_coords(pose_landmarks[end_idx], image_w, image_h)
                    color = LEFT_COLOR if start_idx in POSE_LANDMARKS_LEFT and end_idx in POSE_LANDMARKS_LEFT else RIGHT_COLOR if start_idx in POSE_LANDMARKS_RIGHT and end_idx in POSE_LANDMARKS_RIGHT else MIDDLE_COLOR
                    cv2.line(black_frame, start_xy, end_xy, color, 3)
                draw_custom_skeleton(pose_landmarks, image_w, image_h, black_frame)

            draw_hand(results.left_hand_landmarks, image_w, image_h, black_frame, LEFT_COLOR)
            draw_hand(results.right_hand_landmarks, image_w, image_h, black_frame, RIGHT_COLOR)

            processed_frames.append(black_frame)
    return processed_frames

def extract_mobilenet_features(frames):
    features = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        x = preprocess_input(np.array(resized, dtype=np.float32))
        x = np.expand_dims(x, axis=0)
        feat = mobilenet.predict(x, verbose=0)
        features.append(feat[0])
    return np.array(features)

input_mode = sys.argv[1]
video_path = sys.argv[1]
model_path = "./transformer_v2.keras"
label_map_json = "label_map_tamil.json"

def predict_from_camera_or_video(mode, video_path, model_path, label_map_path=None, max_len=300):
    import json
    
    if mode == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[INFO] Webcam not available. Switching to video file.")
            cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(video_path)

    model = tf.keras.models.load_model(model_path)

    frame_buffer = []

    if label_map_path:
        with open(label_map_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
    else:
        label_map = {}


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if mode == "webcam":
            frame = cv2.flip(frame, 1)

        frame_buffer.append(frame)

        if len(frame_buffer) == 30:  # Collect 30 frames, then predict
            skeleton_frames = extract_mediapipe_keypoints(frame_buffer)
            features = extract_mobilenet_features(skeleton_frames)

            if features.shape[0] < max_len:
                features = np.pad(features, ((0, max_len - features.shape[0]), (0, 0)), mode='constant')
            else:
                features = features[:max_len]

            features_input = np.expand_dims(features, axis=0)
            prediction = model.predict(features_input, verbose=0)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)
            predicted_label = label_map.get(str(predicted_class), f"Class {predicted_class}")

            cv2.putText(frame, f"{predicted_label} ({confidence:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            frame_buffer = []  # Reset buffer for next prediction

        # cv2.imshow("Gesture Recognition", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    print( predicted_label)        
    cap.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_from_camera_or_video(input_mode, video_path, model_path, label_map_json)
