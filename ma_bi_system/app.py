import psycopg2
import face_recognition
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import json
import base64
from PIL import Image

app = Flask(__name__)

# --- DATABASE CONNECTION ---
DB_CONFIG = {
    'dbname': 'ma_bi_system',
    'user': 'postgres',
    'password': 'YOUR_PASSWORD_HERE', # <--- Verify Password
    'host': 'localhost',
    'port': '5432'
}

# --- TUNING ---
BLUR_THRESHOLD = 100  # Lower = allows blurrier images. 100 is standard.

def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    return conn

# --- HELPER: Process Image (Robust) ---
def process_image(file_storage):
    # 1. Open with Pillow to handle weird formats (WEBP, PNG, etc)
    pil_image = Image.open(file_storage).convert('RGB')
    
    # 2. Convert to Numpy Array ensuring 8-bit format (Fixes "Unsupported Image Type")
    img_rgb = np.array(pil_image, dtype=np.uint8)
    
    # 3. Create BGR copy for OpenCV drawing
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    return img_rgb, img_bgr

# --- FILTER: BLUR CHECK ---
def is_image_blurry(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < BLUR_THRESHOLD

# --- HELPER: Draw Face Landmarks ---
def draw_face_landmarks(img_bgr, face_locations):
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(img_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
    return img_bgr

# --- DUPLICATE CHECK ---
def check_if_face_exists(new_encoding):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT full_name, biometric_template FROM users')
    rows = cur.fetchall()
    cur.close()
    conn.close()

    for row in rows:
        db_name = row[0]
        db_encoding = np.array(json.loads(row[1]))
        match = face_recognition.compare_faces([db_encoding], new_encoding, tolerance=0.6)
        if match[0]:
            return True, db_name
    return False, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    try:
        name = request.form['name']
        file = request.files['image']

        img_rgb, img_bgr = process_image(file)

        # 1. BLUR CHECK
        if is_image_blurry(img_bgr):
            return jsonify({'status': 'error', 'message': 'REJECTED: Image is too blurry. Please stand still.'})

        # 2. DETECT FACES
        locations = face_recognition.face_locations(img_rgb)
        
        # 3. GROUP / NO FACE CHECK
        if len(locations) == 0:
            return jsonify({'status': 'error', 'message': 'REJECTED: No face found.'})
        if len(locations) > 1:
            return jsonify({'status': 'error', 'message': f'REJECTED: {len(locations)} faces detected. One person only!'})

        encodings = face_recognition.face_encodings(img_rgb, locations)
        
        # 4. DUPLICATE CHECK
        exists, existing_name = check_if_face_exists(encodings[0])
        if exists:
            return jsonify({'status': 'error', 'message': f'Security Alert: Face already registered to "{existing_name}".'})

        # 5. SAVE
        encoding_json = json.dumps(encodings[0].tolist())
        
        # Save resized image
        img_small = cv2.resize(img_bgr, (300, 300))
        _, buffer = cv2.imencode('.jpg', img_small)
        img_str = base64.b64encode(buffer).decode('utf-8')

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('INSERT INTO users (full_name, biometric_template, original_image) VALUES (%s, %s, %s)', (name, encoding_json, img_str))
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'status': 'success', 'message': f'User {name} registered successfully!'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/identify', methods=['POST'])
def identify():
    try:
        file = request.files['image']
        img_rgb, img_bgr = process_image(file)

        # Blur Check
        if is_image_blurry(img_bgr):
            return jsonify({'status': 'error', 'message': 'Image is too blurry for identification.'})

        locations = face_recognition.face_locations(img_rgb)
        
        if len(locations) == 0: return jsonify({'status': 'error', 'message': 'No face detected.'})
        if len(locations) > 1: return jsonify({'status': 'error', 'message': 'Multiple faces detected. One person only.'})

        encodings = face_recognition.face_encodings(img_rgb, locations)
        unknown_encoding = encodings[0]

        # Draw box on Input
        img_bgr = draw_face_landmarks(img_bgr, locations)

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT full_name, biometric_template, original_image FROM users')
        rows = cur.fetchall()
        cur.close()
        conn.close()

        found_name = "Unknown"
        match_img_b64 = None
        
        for row in rows:
            db_name = row[0]
            db_encoding = np.array(json.loads(row[1]))
            db_img_str = row[2]

            score = face_recognition.face_distance([db_encoding], unknown_encoding)[0]
            score = round(score, 2)

            if score < 0.6: # Match
                found_name = db_name
                if db_img_str:
                    nparr = np.frombuffer(base64.b64decode(db_img_str), np.uint8)
                    db_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    h, w, _ = db_img.shape
                    img_input_resized = cv2.resize(img_bgr, (w, h))
                    combined_img = cv2.hconcat([img_input_resized, db_img])
                    
                    _, buffer = cv2.imencode('.jpg', combined_img)
                    match_img_b64 = base64.b64encode(buffer).decode('utf-8')
                break

        if found_name != "Unknown":
            return jsonify({'status': 'success', 'message': f'FACE VERIFIED: {found_name}', 'score': f'Difference: {score} (Lower is better)', 'image_data': match_img_b64})
        else:
            return jsonify({'status': 'fail', 'message': 'Access Denied: Face not recognized.'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/verify_manual', methods=['POST'])
def verify_manual():
    try:
        file1 = request.files['image1']
        file2 = request.files['image2']

        img1_rgb, img1_bgr = process_image(file1)
        img2_rgb, img2_bgr = process_image(file2)

        # Gatekeepers for Manual Mode
        if is_image_blurry(img1_bgr): return jsonify({'status': 'error', 'message': 'Face A is too blurry.'})
        if is_image_blurry(img2_bgr): return jsonify({'status': 'error', 'message': 'Face B is too blurry.'})

        loc1 = face_recognition.face_locations(img1_rgb)
        loc2 = face_recognition.face_locations(img2_rgb)

        if len(loc1) != 1: return jsonify({'status': 'error', 'message': 'Face A: Must contain exactly one face.'})
        if len(loc2) != 1: return jsonify({'status': 'error', 'message': 'Face B: Must contain exactly one face.'})

        enc1 = face_recognition.face_encodings(img1_rgb, loc1)
        enc2 = face_recognition.face_encodings(img2_rgb, loc2)

        # Draw Boxes
        img1_bgr = draw_face_landmarks(img1_bgr, loc1)
        img2_bgr = draw_face_landmarks(img2_bgr, loc2)

        # Compare
        distance = face_recognition.face_distance([enc1[0]], enc2[0])[0]
        score = round(distance, 2)
        is_match = distance < 0.6

        # Side-by-Side
        h1, w1, _ = img1_bgr.shape
        h2, w2, _ = img2_bgr.shape
        target_h = min(h1, h2)
        
        img1_res = cv2.resize(img1_bgr, (int(w1 * target_h / h1), target_h))
        img2_res = cv2.resize(img2_bgr, (int(w2 * target_h / h2), target_h))
        
        combined = cv2.hconcat([img1_res, img2_res])
        _, buffer = cv2.imencode('.jpg', combined)
        result_b64 = base64.b64encode(buffer).decode('utf-8')

        if is_match:
            return jsonify({'status': 'success', 'message': 'POSITIVE MATCH', 'score': f'Distance: {score}', 'image_data': result_b64})
        else:
            return jsonify({'status': 'fail', 'message': 'NEGATIVE MATCH', 'score': f'Distance: {score}', 'image_data': result_b64})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
