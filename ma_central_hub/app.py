from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Runs on Port 8080 (The standard "Dashboard" port)
    app.run(debug=True, port=8080)