from flask import Flask, request, render_template, redirect, url_for, Response, send_from_directory, make_response
import cv2
import urllib.request
import numpy as np
import time
from ultralytics import YOLO, YOLOv10
import os
import keras
from PIL import Image
import cvzone
import math
import mysql.connector
from io import BytesIO
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load models
model_path = os.path.join(os.getcwd(), 'models', 'fire.pt')
yolov10_model = YOLOv10(model_path)
vgg16_model = keras.models.load_model('models/my_keras_VGG16_model.h5')
fire_model = YOLO('models/fire_1.pt')

# ESP32-CAM URL
url = "http://192.168.88.109:81/stream"

# MySQL database connection
db_config = {
    'user': 'root',
    'password': 'Huy2582002',
    'host': 'localhost',
    'database': 'fire_detection'
}


def connect_to_db():
    return mysql.connector.connect(**db_config)


def save_frame_to_db(frame):
    conn = connect_to_db()
    cursor = conn.cursor()
    frame_byte_arr = cv2.imencode('.jpg', frame)[1].tobytes()
    cursor.execute("INSERT INTO detected_frames (frame) VALUES (%s)", (frame_byte_arr,))
    conn.commit()
    cursor.close()
    conn.close()


def save_video_upload_info(original_filename, output_filename):
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO video_uploads (original_filename, output_filename) VALUES (%s, %s)",
                   (original_filename, output_filename))
    conn.commit()
    cursor.close()
    conn.close()


def save_image_upload_info(filename, predicted_label, image_data):
    true_label = ''.join([i for i in filename if not i.isdigit()]).split('.')[0]
    conn = connect_to_db()
    cursor = conn.cursor()
    # frame_byte_arr = cv2.imencode('.jpg', image_data)[1].tobytes()
    cursor.execute("INSERT INTO image_uploads (filename, predicted_label, true_label, image_blob) VALUES (%s, %s, %s, %s)",
                   (filename, predicted_label, true_label, image_data))
    conn.commit()
    cursor.close()
    conn.close()

def get_frames_from_db():
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, frame FROM detected_frames")
    frames = cursor.fetchall()
    cursor.close()
    conn.close()
    return frames


def preprocess_image(img_path):
    image = Image.open(img_path)
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def detect_fire_image(image_path):
    image = preprocess_image(image_path)
    prediction = vgg16_model.predict(image)
    label = "fire" if prediction[0][0] < 0.5 else "non_fire"
    return prediction


classnames = ['fire']


def detect_fire_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_' + os.path.basename(video_path))
    out = cv2.VideoWriter(out_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = fire_model(frame, stream=True)

        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence > 50:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100], scale=1.5,
                                       thickness=2)
                    #save_frame_to_db(frame)  # Save frame to database if fire is detected

        out.write(frame)

    cap.release()
    out.release()

    save_video_upload_info(os.path.basename(video_path),
                           os.path.basename(out_path))  # Save upload information to the database

    return out_path


def generate_frames():
    bytes_stream = b""

    while True:
        try:
            with urllib.request.urlopen(url) as stream:
                while True:
                    bytes_stream += stream.read(1024)
                    a = bytes_stream.find(b"\xff\xd8")  # Start of JPEG
                    b = bytes_stream.find(b"\xff\xd9")  # End of JPEG

                    if a != -1 and b != -1:
                        jpg = bytes_stream[a: b + 2]
                        bytes_stream = bytes_stream[b + 2:]

                        if len(jpg) == 0:
                            print("Error: Empty JPEG buffer")
                            continue

                        img_np = np.frombuffer(jpg, dtype=np.uint8)
                        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

                        if img is None:
                            print("Error: Failed to decode image")
                            continue

                        # Fire detection
                        results = yolov10_model(img, conf=0.8)
                        # Draw bounding boxes
                        for result in results:
                            boxes = result.boxes
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                                c = box.cls
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                label = yolov10_model.names[int(c)]
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(
                                    img,
                                    label,
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9,
                                    (0, 255, 0),
                                    2,
                                )
                                save_frame_to_db(img)  # Save frame to database if fire is detected

                        ret, buffer = cv2.imencode('.jpg', img)
                        if not ret:
                            print("Error: Failed to encode image")
                            continue

                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except urllib.error.HTTPError as e:
            print(f"HTTP Error: {e.code} - {e.reason}")
            time.sleep(1)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/inference', methods=['GET', 'POST'])
def inference():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        if file.content_type.startswith('image/'):
            prediction = detect_fire_image(file_path)
            label = "fire" if prediction[0][0] < 0.5 else "non_fire"

            # Read image file and convert to blob data
            with open(file_path, 'rb') as f:
                image_data = f.read()

            # Save image information and blob data to the database
            save_image_upload_info(file.filename, label, image_data)

            return render_template('result.html', label=label, file_path=file_path)
        elif file.content_type.startswith('video/'):
            out_path = detect_fire_video(file_path)
            video_filename = 'uploads/output_' + file.filename
            return render_template('result.html', video_path=video_filename)
    return render_template('inference.html')


ALLOWED_EXTENSION = ['mp4']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION


@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return 'No video file found'

    video = request.files['video']

    if video.filename == '':
        return 'No video file selected'

    if video and allowed_file(video.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        video.save(file_path)  # Save the uploaded video to the server

        out_path = detect_fire_video(file_path)
        video_filename = os.path.basename(out_path)  # Get the base name of the output file

        return render_template('download.html', video_name=video_filename)

    return 'Invalid file type'


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/camera')
def camera():
    return render_template('camera.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detected_frames')
def detected_frames():
    frames = get_frames_from_db()
    encoded_frames = [(id, timestamp, base64.b64encode(frame).decode('utf-8')) for id, timestamp, frame in frames]
    return render_template('detected_frames.html', frames=encoded_frames)


@app.route('/uploads/images')
def view_images():
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT filename, predicted_label, true_label, timestamp, image_blob FROM image_uploads")
    images = cursor.fetchall()
    cursor.close()
    conn.close()

    decoded_images = []
    for filename, predicted_label, true_label, timestamp, image_blob in images:
        encoded_image = base64.b64encode(image_blob).decode('utf-8')
        decoded_images.append((filename, predicted_label, true_label, timestamp, encoded_image))

    return render_template('view_images.html', images=decoded_images)


@app.route('/uploads/videos')
def view_videos():
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT original_filename, output_filename, timestamp FROM video_uploads")
    videos = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('view_videos.html', videos=videos)


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True, host='0.0.0.0')
