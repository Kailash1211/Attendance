import os
import cv2
import numpy as np
import face_recognition
import mysql.connector
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MARKED_FOLDER'] = 'marked'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MARKED_FOLDER'], exist_ok=True)

# Load known faces
KNOWN_FACES_DIR = "known_faces"
known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            filename = image_file.filename
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(upload_path)

            image_to_scan = cv2.imread(upload_path)
            if image_to_scan is None:
                return "Error loading image", 400

            rgb_image = cv2.cvtColor(image_to_scan, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

            detected_names = set()

            for face_encoding, face_location in zip(face_encodings, face_locations):
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None
                name = "Unknown"
                if best_match_index is not None and face_distances[best_match_index] < 0.6:
                    name = known_face_names[best_match_index]
                detected_names.add(name)

                top, right, bottom, left = face_location
                cv2.rectangle(image_to_scan, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(image_to_scan, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            output_filename = f"marked_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            output_path = os.path.join(app.config['MARKED_FOLDER'], output_filename)
            cv2.imwrite(output_path, image_to_scan)

            # Save to MySQL
            date_today = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            try:
                db = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="root123",
                    database="attendance_01"
                )
                cursor = db.cursor()
                query = "INSERT INTO attendance (name, timestamp) VALUES (%s, %s)"
                values = [(name, date_today) for name in detected_names if name != "Unknown"]
                if values:
                    cursor.executemany(query, values)
                    db.commit()
                cursor.close()
                db.close()
            except Exception as e:
                print("Database error:", e)

            # Save to Excel
            df = pd.DataFrame({'Name': list(detected_names), 'Timestamp': [date_today] * len(detected_names)})
            excel_filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            excel_path = os.path.join("marked", excel_filename)
            df.to_excel(excel_path, index=False)

            return render_template("index.html", names=detected_names, marked_img=output_filename, excel_file=excel_filename)

    return render_template("index.html", names=[], marked_img=None, excel_file=None)

@app.route('/marked/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['MARKED_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)