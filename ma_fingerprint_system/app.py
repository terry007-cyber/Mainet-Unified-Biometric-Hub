import psycopg2
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import json
import base64

app = Flask(__name__)

# --- DATABASE CONNECTION ---
DB_CONFIG = {
    'dbname': 'ma_bi_system',
    'user': 'postgres',
    'password': 'YOUR_PASSWORD_HERE', # <--- Verify Password
    'host': 'localhost',
    'port': '5432'
}

# --- TUNING PARAMETERS ---
MATCH_THRESHOLD = 15       # How many lines must connect to say "MATCH"
MIN_KEYPOINTS_REQUIRED = 40 # Minimum dots to say "REAL FINGERPRINT"
BLUR_THRESHOLD = 150       # Low score = Blurry

def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    return conn

def process_image(file_storage):
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    return img

# --- FILTERS (The Gatekeepers) ---
def is_image_blurry(img):
    variance = cv2.Laplacian(img, cv2.CV_64F).var()
    return variance < BLUR_THRESHOLD

def contains_human_face(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    return len(faces) > 0

# --- DUPLICATE CHECK ---
def check_if_fingerprint_exists(new_descriptors):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT full_name, fingerprint_template FROM fingerprints')
    rows = cur.fetchall()
    cur.close()
    conn.close()

    for row in rows:
        db_desc = np.array(json.loads(row[1]), dtype=np.float32)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(new_descriptors, db_desc, k=2)
        good_points = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_points.append(m)
        if len(good_points) >= MATCH_THRESHOLD:
            return True, row[0]
    return False, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    try:
        name = request.form['name']
        file = request.files['image']
        img = process_image(file)
        if img is None: return jsonify({'status': 'error', 'message': 'Invalid image.'})

        # Gatekeepers
        if is_image_blurry(img): return jsonify({'status': 'error', 'message': 'REJECTED: Image is too blurry.'})
        if contains_human_face(img): return jsonify({'status': 'error', 'message': 'REJECTED: Face detected. Please upload a FINGERPRINT.'})

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)

        if descriptors is None or len(keypoints) < MIN_KEYPOINTS_REQUIRED:
            return jsonify({'status': 'error', 'message': 'REJECTED: Not enough details. Is this a valid fingerprint?'})

        exists, existing_name = check_if_fingerprint_exists(descriptors)
        if exists:
            return jsonify({'status': 'error', 'message': f'Security Alert: Already registered to "{existing_name}".'})

        desc_json = json.dumps(descriptors.tolist())
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode('utf-8')

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('INSERT INTO fingerprints (full_name, fingerprint_template, original_image) VALUES (%s, %s, %s)', (name, desc_json, img_str))
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'status': 'success', 'message': f'Registered {name} successfully!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/identify', methods=['POST'])
def identify():
    try:
        file = request.files['image']
        img_input = process_image(file)
        if img_input is None: return jsonify({'status': 'error', 'message': 'Invalid image.'})

        if is_image_blurry(img_input): return jsonify({'status': 'error', 'message': 'Image is too blurry.'})
        
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img_input, None)
        
        if des1 is None or len(kp1) < MIN_KEYPOINTS_REQUIRED:
             return jsonify({'status': 'error', 'message': 'Not a valid fingerprint.'})

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT full_name, fingerprint_template, original_image FROM fingerprints')
        rows = cur.fetchall()
        cur.close()
        conn.close()

        best_score = 0
        best_match_name = "Unknown"
        best_match_img_db = None
        best_kp2 = None
        best_good_matches = []

        for row in rows:
            db_name = row[0]
            db_desc = np.array(json.loads(row[1]), dtype=np.float32)
            db_img_str = row[2] 
            if not db_img_str: continue

            nparr = np.frombuffer(base64.b64decode(db_img_str), np.uint8)
            img_db = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            kp2, _ = sift.detectAndCompute(img_db, None)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, db_desc, k=2)

            good_points = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_points.append(m)

            if len(good_points) > best_score:
                best_score = len(good_points)
                if best_score >= MATCH_THRESHOLD:
                    best_match_name = db_name
                    best_match_img_db = img_db
                    best_kp2 = kp2
                    best_good_matches = good_points

        result_image_b64 = None
        if best_match_name != "Unknown" and best_match_img_db is not None:
            match_img = cv2.drawMatches(img_input, kp1, best_match_img_db, best_kp2, best_good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0, 255, 0))
            _, buffer = cv2.imencode('.jpg', match_img)
            result_image_b64 = base64.b64encode(buffer).decode('utf-8')

        if best_match_name != "Unknown":
            return jsonify({'status': 'success', 'message': f'MATCH FOUND: {best_match_name}', 'score': f'Score: {best_score}', 'image_data': result_image_b64})
        else:
            return jsonify({'status': 'fail', 'message': f'No Match Found. (Score: {best_score})'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# --- UPDATED MANUAL VERIFICATION (With Gatekeepers) ---
@app.route('/verify_manual', methods=['POST'])
def verify_manual():
    try:
        file1 = request.files['image1']
        file2 = request.files['image2']

        img1 = process_image(file1)
        img2 = process_image(file2)

        if img1 is None or img2 is None:
            return jsonify({'status': 'error', 'message': 'Invalid image files.'})

        # --- GATEKEEPER CHECKS (Apply to BOTH images) ---
        
        # Check Image A
        if is_image_blurry(img1): return jsonify({'status': 'error', 'message': 'Image A is too blurry.'})
        if contains_human_face(img1): return jsonify({'status': 'error', 'message': 'Image A contains a FACE, not a fingerprint.'})

        # Check Image B
        if is_image_blurry(img2): return jsonify({'status': 'error', 'message': 'Image B is too blurry.'})
        if contains_human_face(img2): return jsonify({'status': 'error', 'message': 'Image B contains a FACE, not a fingerprint.'})

        # SIFT Analysis
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # Detail Check (Are they actually fingerprints?)
        if des1 is None or len(kp1) < MIN_KEYPOINTS_REQUIRED:
            return jsonify({'status': 'error', 'message': 'Image A does not have enough details to be a fingerprint.'})
        if des2 is None or len(kp2) < MIN_KEYPOINTS_REQUIRED:
            return jsonify({'status': 'error', 'message': 'Image B does not have enough details to be a fingerprint.'})

        # Matching Logic
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good_points = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_points.append(m)

        score = len(good_points)
        is_match = score >= MATCH_THRESHOLD

        match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_points, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0, 255, 0))
        _, buffer = cv2.imencode('.jpg', match_img)
        result_image_b64 = base64.b64encode(buffer).decode('utf-8')

        if is_match:
            return jsonify({'status': 'success', 'message': 'POSITIVE MATCH CONFIRMED', 'score': f'Score: {score}', 'image_data': result_image_b64})
        else:
            return jsonify({'status': 'fail', 'message': 'NEGATIVE. Images do not match.', 'score': f'Score: {score}', 'image_data': result_image_b64})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
