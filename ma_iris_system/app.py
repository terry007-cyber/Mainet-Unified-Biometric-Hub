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

# --- TUNING ---
MATCH_THRESHOLD = 20       # Iris needs more points than fingerprint
MIN_KEYPOINTS_REQUIRED = 15 
BLUR_THRESHOLD = 100       

def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    return conn

def process_image(file_storage):
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    return img

# --- FILTERS ---
def is_image_blurry(img):
    variance = cv2.Laplacian(img, cv2.CV_64F).var()
    return variance < BLUR_THRESHOLD

def validate_is_eye(img):
    # Load OpenCV's built-in Eye Detector
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    return len(eyes) > 0

# --- DUPLICATE CHECK ---
def check_if_iris_exists(new_descriptors):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT full_name, iris_template FROM iris')
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
        
        # CRITICAL: Ensure it is an Eye
        if not validate_is_eye(img): return jsonify({'status': 'error', 'message': 'REJECTED: No Human Eye detected. Don\'t try to spoof me!'})

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)

        if descriptors is None or len(keypoints) < MIN_KEYPOINTS_REQUIRED:
            return jsonify({'status': 'error', 'message': 'REJECTED: Not enough texture details.'})

        # Check Duplicates
        exists, existing_name = check_if_iris_exists(descriptors)
        if exists:
            return jsonify({'status': 'error', 'message': f'Security Alert: This Iris is already registered to "{existing_name}".'})

        desc_json = json.dumps(descriptors.tolist())
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode('utf-8')

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('INSERT INTO iris (full_name, iris_template, original_image) VALUES (%s, %s, %s)', (name, desc_json, img_str))
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'status': 'success', 'message': f'Iris Scan for {name} registered!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/identify', methods=['POST'])
def identify():
    try:
        file = request.files['image']
        img_input = process_image(file)
        if img_input is None: return jsonify({'status': 'error', 'message': 'Invalid image.'})

        if is_image_blurry(img_input): return jsonify({'status': 'error', 'message': 'Image is too blurry.'})
        if not validate_is_eye(img_input): return jsonify({'status': 'error', 'message': 'REJECTED: Not an eye.'})
        
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img_input, None)
        
        if des1 is None: return jsonify({'status': 'error', 'message': 'No details found.'})

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT full_name, iris_template, original_image FROM iris')
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
            return jsonify({'status': 'success', 'message': f'IRIS VERIFIED: {best_match_name}', 'score': f'Matches: {best_score}', 'image_data': result_image_b64})
        else:
            return jsonify({'status': 'fail', 'message': f'No Match Found. (Score: {best_score})'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# --- ROUTE: MANUAL COMPARATOR (1:1) ---
@app.route('/verify_manual', methods=['POST'])
def verify_manual():
    try:
        file1 = request.files['image1']
        file2 = request.files['image2']

        img1 = process_image(file1)
        img2 = process_image(file2)

        # Gatekeepers for Manual Mode
        if not validate_is_eye(img1): return jsonify({'status': 'error', 'message': 'Image A is NOT an eye.'})
        if not validate_is_eye(img2): return jsonify({'status': 'error', 'message': 'Image B is NOT an eye.'})

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return jsonify({'status': 'error', 'message': 'Not enough detail.'})

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
            return jsonify({'status': 'success', 'message': 'POSITIVE MATCH', 'score': f'Score: {score}', 'image_data': result_image_b64})
        else:
            return jsonify({'status': 'fail', 'message': 'NEGATIVE MATCH', 'score': f'Score: {score}', 'image_data': result_image_b64})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5002)
