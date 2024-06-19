# app.py
from flask import Flask, render_template, request, redirect, url_for, session, Response, send_from_directory
from flask_socketio import SocketIO, emit
import cv2
import pandas as pd 
from ultralytics import YOLO 
import numpy as np
import pytesseract
from datetime import datetime
import time
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

model = YOLO('best.pt')

app = Flask(__name__)
app.secret_key = '11223344' 
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins='*', engineio_logger=True)

users = {'admin': 'password'}

def generate_frames(video_path, side):
    cap = cv2.VideoCapture(video_path)

    my_file = open("coco1.txt", "r")
    data = my_file.read()
    class_list = data.split("\n") 

    area = [(27, 417), (16, 456), (1015, 451), (992, 417)]

    count = 0
    processed_numbers = set()

    while True:    
        ret, frame = cap.read()
        count += 1
        if count % 3 != 0:
            continue
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])

            d = int(row[5])
            c = class_list[d]
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2
            result = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
            if result >= 0:
                crop = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 10, 20, 20)

                text = pytesseract.image_to_string(gray).strip()
                text = text.replace('(', '').replace(')', '').replace(',', '').replace(']','')
                if text not in processed_numbers:
                    processed_numbers.add(text) 
                    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open("car_plate_data.txt", "a") as file:
                        file.write(f"{text}\t{current_datetime}\n")
                    socketio.emit('update', {'camera': side, 'text': text, 'timestamp': current_datetime})

        cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    if 'username' in session:
        return render_template('page1.html')  
    else:
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('live_page')) 
        else:
            return render_template('login.html', error='Invalid username or password')
    else:
        return render_template('login.html', error=None)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))  

@app.route('/video_feed/<page>/<side>')
def video_feed(page, side):
    if page == 'page2':
        if side == 'left':
            return Response(generate_frames('videos/demo.mp4', side), mimetype='multipart/x-mixed-replace; boundary=frame')
        elif side == 'right':
            return Response(generate_frames('videos/demo.mp4', side), mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            return 'Invalid side'
    elif page == 'page3':
        if side == 'left':
            return Response(generate_frames('videos/demo.mp4', side), mimetype='multipart/x-mixed-replace; boundary=frame')
        elif side == 'right':
            return Response(generate_frames('videos/demo2.mp4', side), mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            return 'Invalid side'
    else:
        return 'Invalid page'

@app.route('/live')
def live_page():
    return render_template('page1.html')

@app.route('/main_gate')
def main_gate_page():
    return render_template('page2.html')

@app.route('/parking')
def parking_page():
    return render_template('page3.html')

if __name__ == "__main__":
    socketio.run(app, debug=True)
