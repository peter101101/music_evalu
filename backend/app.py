# backend/app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import librosa
import numpy as np
import os
from pydub import AudioSegment
import sqlite3

app = Flask(__name__)
CORS(app)

# 创建 uploads 文件夹
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# 初始化数据库
def init_db():
    conn = sqlite3.connect('audio_analysis.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pitch REAL,
            tempo REAL,
            spectral_centroid REAL,
            zcr REAL,
            mfcc TEXT,
            pitch_comment TEXT,
            tempo_comment TEXT,
            spectral_comment TEXT,
            zcr_comment TEXT,
            pitch_comparison TEXT,
            tempo_comparison TEXT,
            spectral_comparison TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/evaluate', methods=['POST'])
def evaluate_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    
    # Check file extension
    if not file.filename.endswith(('.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aiff')):
        return jsonify({'error': 'Unsupported file format. Please upload a WAV, MP3, OGG, FLAC, AIFF, or M4A file.'}), 400

    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Convert to WAV if necessary
    if not file.filename.endswith('.wav'):
        audio = AudioSegment.from_file(file_path)
        wav_file_path = file_path.rsplit('.', 1)[0] + '.wav'
        audio.export(wav_file_path, format='wav')
        file_path = wav_file_path  # 更新文件路径为转换后的 WAV 文件路径

    # Perform audio analysis
    report = analyze_audio(file_path)

    # Delete the uploaded file
    os.remove(file_path)

    # Save results to database
    save_to_db(report)

    return jsonify(report), 200

def analyze_audio(file_path):
    try:
        print("Loading audio file with librosa...")
        y, sr = librosa.load(file_path, sr=None)
        print(f"Audio file loaded successfully with {len(y)} samples.")

        # 使用 librosa 提取特征
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch = np.mean(pitches)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)

        report = generate_report(pitch, tempo, spectral_centroid, zcr, mfcc)
        return report
    except Exception as e:
        print(f"Error during audio analysis: {e}")
        return {'error': str(e)}

def generate_report(pitch, tempo, spectral_centroid, zcr, mfcc):
    pitch_comment = get_pitch_comment(float(pitch))
    tempo_comment = get_tempo_comment(float(tempo))
    spectral_comment = get_spectral_comment(float(spectral_centroid))
    zcr_comment = get_zcr_comment(float(zcr))
    suitable_songs = get_suitable_songs(float(pitch), float(tempo))

    # 假设我们有一些专业歌手的基准数据
    professional_pitch = 220  # 例如，专业歌手的平均音高
    professional_tempo = 120  # 例如，专业歌手的平均节奏
    professional_spectral_centroid = 2500  # 例如，专业歌手的平均声色

    pitch_comparison = compare_to_professional(float(pitch), professional_pitch)
    tempo_comparison = compare_to_professional(float(tempo), professional_tempo)
    spectral_comparison = compare_to_professional(float(spectral_centroid), professional_spectral_centroid)

    improvement_suggestions = {
        'pitch': "尝试练习音阶和音准，使用钢琴或调音器辅助。",
        'tempo': "使用节拍器练习，保持稳定的节奏感。",
        'spectral_centroid': "如果声色过亮，尝试放松喉咙，增加共鸣；如果过暗，尝试提高音调，增加亮度。",
        'zcr': "如果零交叉率过高，尝试减少噪音和不必要的音频变化。"
    }

    return {
        'pitch': float(pitch),
        'tempo': float(tempo),
        'spectral_centroid': float(spectral_centroid),
        'zcr': float(zcr),
        'mfcc': [float(m) for m in mfcc],  # 转换为标准 float 类型
        'recommendations': suitable_songs,
        'comments': {
            'pitch_comment': pitch_comment,
            'tempo_comment': tempo_comment,
            'spectral_comment': spectral_comment,
            'zcr_comment': zcr_comment,
            'pitch_comparison': pitch_comparison,
            'tempo_comparison': tempo_comparison,
            'spectral_comparison': spectral_comparison
        },
        'professional': {
            'pitch': professional_pitch,
            'tempo': professional_tempo,
            'spectral_centroid': professional_spectral_centroid
        },
        'improvement_suggestions': improvement_suggestions
    }

def compare_to_professional(value, professional_value):
    if value < professional_value * 0.9:
        return "低于专业水平"
    elif professional_value * 0.9 <= value <= professional_value * 1.1:
        return "接近专业水平"
    else:
        return "高于专业水平"

def get_pitch_comment(pitch):
    if pitch < 110:
        return "音高较低，可能不适合大多数歌曲。"
    elif 110 <= pitch < 440:
        return "音高适中，适合多种歌曲。"
    else:
        return "音高较高，适合高音歌曲。"

def get_tempo_comment(tempo):
    if tempo < 60:
        return "节奏较慢，适合抒情歌曲。"
    elif 60 <= tempo < 120:
        return "节奏适中，适合多种类型的音乐。"
    else:
        return "节奏较快，适合舞曲和流行音乐。"

def get_spectral_comment(spectral_centroid):
    if spectral_centroid < 2000:
        return "声色较暗，适合低音歌曲。"
    else:
        return "声色较亮，适合高音歌曲。"

def get_zcr_comment(zcr):
    if zcr < 0.1:
        return "零交叉率较低，音频较为平稳。"
    elif 0.1 <= zcr < 0.3:
        return "零交叉率适中，音频变化适中。"
    else:
        return "零交叉率较高，音频变化较大。"

def get_suitable_songs(pitch, tempo):
    # 这里可以实现更复杂的推荐算法
    if pitch < 200 and tempo < 100:
        return ['Song A', 'Song B']
    elif 200 <= pitch < 300 and 100 <= tempo < 120:
        return ['Song C', 'Song D']
    else:
        return ['Song E', 'Song F']

def save_to_db(report):
    conn = sqlite3.connect('audio_analysis.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO analysis_results (
            pitch, tempo, spectral_centroid, zcr, mfcc,
            pitch_comment, tempo_comment, spectral_comment, zcr_comment,
            pitch_comparison, tempo_comparison, spectral_comparison
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        report['pitch'], report['tempo'], report['spectral_centroid'], report['zcr'], str(report['mfcc']),
        report['comments']['pitch_comment'], report['comments']['tempo_comment'],
        report['comments']['spectral_comment'], report['comments']['zcr_comment'],
        report['comments']['pitch_comparison'], report['comments']['tempo_comparison'],
        report['comments']['spectral_comparison']
    ))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    app.run(debug=True)