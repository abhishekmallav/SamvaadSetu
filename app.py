from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import ffmpeg
import time
import glob
import json
import requests
import nltk
from nltk.tokenize import sent_tokenize
import language_tool_python
import google.generativeai as genai
import markdown
import re
nltk.download('punkt_tab')
# Download NLTK data (only needed first time)


def download_nltk_data():
    print("📚 [DEBUG] Checking NLTK data...")
    try:
        nltk.data.find('tokenizers/punkt')
        print("✅ [DEBUG] NLTK punkt tokenizer already installed")
    except LookupError:
        print("📥 [INFO] Downloading punkt data for NLTK...")
        nltk.download('punkt_tab')
        print("✅ [DEBUG] NLTK punkt tokenizer installed successfully")


# Call the function before Flask app initialization
download_nltk_data()

app = Flask(__name__)

# Define folders
UPLOAD_FOLDER = "videos"
REPORTS_FOLDER = "reports"
POSTURE_FOLDER = "posture"
TRANSCRIPTS_FOLDER = "transcripts"
ANALYSIS_FOLDER = "analysis"
LLM_FOLDER = "LLM"

# Create folders if they don't exist
print("📁 [DEBUG] Creating necessary folders...")
for folder in [UPLOAD_FOLDER, REPORTS_FOLDER, POSTURE_FOLDER, TRANSCRIPTS_FOLDER, ANALYSIS_FOLDER, LLM_FOLDER]:
    os.makedirs(folder, exist_ok=True)
print("✅ [DEBUG] Folders created successfully")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Deepgram API settings
DEEPGRAM_API_KEY = "API KEY"
DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"

# Configure Gemini API
GEMINI_API_KEY = "API KEY"
genai.configure(api_key=GEMINI_API_KEY)

# Initialize language tool for grammar checking
language_tool = None


def init_language_tool():
    global language_tool
    print("🛠️ [DEBUG] Initializing LanguageTool...")
    if language_tool is None:
        try:
            language_tool = language_tool_python.LanguageTool('en-US')
            print("✅ [DEBUG] LanguageTool initialized successfully")
        except Exception as e:
            print(f"❌ [ERROR] Failed to initialize LanguageTool: {e}")
            language_tool = None


init_language_tool()

# List of common filler words
FILLER_WORDS = [
    "um", "uh", "er", "ah", "like", "you know", "sort of", "kind of",
    "basically", "literally", "actually", "so", "well", "i mean", "right",
    "okay", "hmm"
]


@app.route("/save_video", methods=["POST"])
def save_video():
    print("🎥 [DEBUG] Video recording endpoint /save_video called")
    if "video" not in request.files:
        print("❌ [ERROR] No video file uploaded")
        return "No video file uploaded", 400

    video_file = request.files["video"]
    timestamp = int(time.time())
    webm_path = os.path.join(
        app.config["UPLOAD_FOLDER"], f"recording_{timestamp}.webm")
    mp4_path = os.path.join(
        app.config["UPLOAD_FOLDER"], f"recording_{timestamp}.mp4")
    wav_path = os.path.join(
        app.config["UPLOAD_FOLDER"], f"recording_{timestamp}.wav")
    posture_report_path = os.path.join(
        POSTURE_FOLDER, f"posture_report_{timestamp}.json")
    transcript_path = os.path.join(
        TRANSCRIPTS_FOLDER, f"transcript_{timestamp}.json")
    analysis_path = os.path.join(ANALYSIS_FOLDER, f"analysis_{timestamp}.json")
    combined_report_path = os.path.join(
        REPORTS_FOLDER, f"report_{timestamp}.json")

    print("💾 [DEBUG] Video is being saved as WebM...")
    video_file.save(webm_path)
    print(f"✅ [DEBUG] WebM video saved: {webm_path}")

    if "report" in request.files:
        print("📊 [DEBUG] Posture report received")
        report_file = request.files["report"]
        print("💾 [DEBUG] Saving posture report...")
        report_file.save(posture_report_path)
        print(f"✅ [DEBUG] Posture report saved: {posture_report_path}")

    print("🔄 [DEBUG] Video is converting to MP4...")
    try:
        ffmpeg.input(webm_path).output(mp4_path, vcodec="libx264", acodec="aac").run(
            overwrite_output=True, capture_stdout=True, capture_stderr=True)
        print(f"✅ [DEBUG] Video converted to MP4: {mp4_path}")
    except ffmpeg.Error as e:
        print(f"❌ [ERROR] Video conversion failed: {e.stderr.decode()}")
        return "Video conversion failed", 500

    print("🎵 [DEBUG] Audio is being extracted from the video...")
    try:
        ffmpeg.input(webm_path).audio.output(wav_path, acodec="pcm_s16le", ar=44100, ac=2).run(
            overwrite_output=True, capture_stdout=True, capture_stderr=True)
        print(f"✅ [DEBUG] Audio extracted to WAV: {wav_path}")
    except ffmpeg.Error as e:
        print(f"❌ [ERROR] Audio extraction failed: {e.stderr.decode()}")
        return "Audio extraction failed", 500

    print("🗣️ [DEBUG] Request sent to Deepgram for transcription...")
    try:
        process_audio_with_deepgram(wav_path, transcript_path, analysis_path)
        print("✅ [DEBUG] Deepgram processing completed")
    except Exception as e:
        print(f"❌ [ERROR] Deepgram processing failed: {e}")

    print("📦 [DEBUG] Generating combined report...")
    combined_report = {}
    for path, key in [(posture_report_path, "posture"), (transcript_path, "transcript"), (analysis_path, "analysis")]:
        if os.path.exists(path):
            print(f"📄 [DEBUG] Loading {key} data from {path}")
            with open(path, 'r') as f:
                combined_report[key] = json.load(f)
    print("💾 [DEBUG] Saving combined report...")
    with open(combined_report_path, 'w') as f:
        json.dump(combined_report, f, indent=2)
    print(f"✅ [DEBUG] Combined report saved: {combined_report_path}")

    return jsonify({"message": "Video processed successfully"}), 200


def process_audio_with_deepgram(audio_path, transcript_path, analysis_path):
    print("🚀 [DEBUG] Preparing audio for Deepgram API...")
    with open(audio_path, 'rb') as audio_file:
        audio_data = audio_file.read()
    headers = {'Authorization': f'Token {DEEPGRAM_API_KEY}',
               'Content-Type': 'audio/wav'}
    params = {'punctuate': 'true', 'diarize': 'true',
              'model': 'general', 'tier': 'nova'}
    print("📤 [DEBUG] Sending request to Deepgram API...")
    response = requests.post(
        DEEPGRAM_API_URL, headers=headers, params=params, data=audio_data)
    print(f"📥 [DEBUG] Response received from Deepgram: {response.status_code}")
    if response.status_code == 200:
        transcript_data = response.json()
        print("✅ [DEBUG] Transcript data received from Deepgram")
        print("💾 [DEBUG] Saving transcript...")
        with open(transcript_path, 'w') as f:
            json.dump(transcript_data, f, indent=2)
        print(f"✅ [DEBUG] Transcript saved: {transcript_path}")
        if 'results' in transcript_data and 'channels' in transcript_data['results']:
            print("🔍 [DEBUG] Starting transcript analysis...")
            analyze_transcript(transcript_data, analysis_path)
            print("✅ [DEBUG] Transcript analysis completed")
    else:
        print(
            f"❌ [ERROR] Deepgram API error: {response.status_code} - {response.text}")


def analyze_transcript(transcript_data, analysis_path):
    print("🔍 [DEBUG] Starting transcript analysis")
    channel = transcript_data['results']['channels'][0]
    alternative = channel['alternatives'][0]
    transcript_text = alternative['transcript']
    words = alternative.get('words', [])
    if not transcript_text.strip():
        print("⚠️ [DEBUG] No speech detected in the audio")
        with open(analysis_path, 'w') as f:
            json.dump({"error": "No speech detected"}, f)
        return

    print("⏱️ [DEBUG] Calculating speech duration...")
    total_duration = words[-1]['end'] - words[0]['start'] if words else 0
    print(f"✅ [DEBUG] Speech duration: {total_duration:.2f} seconds")
    print("📚 [DEBUG] Counting words...")
    word_count = len(words)
    print(f"✅ [DEBUG] Word count: {word_count}")
    print("📈 [DEBUG] Calculating words per minute...")
    wpm = int((word_count / total_duration) * 60) if total_duration > 0 else 0
    print(f"✅ [DEBUG] Words per minute: {wpm}")

    print("🗣️ [DEBUG] Detecting filler words...")
    filler_word_counts = {}
    total_filler_words = 0
    for word_obj in words:
        word_text = word_obj['word'].lower()
        for filler in FILLER_WORDS:
            if filler == word_text or (len(filler.split()) > 1 and filler in transcript_text.lower()):
                filler_word_counts[filler] = filler_word_counts.get(
                    filler, 0) + 1
                total_filler_words += 1
    print(f"✅ [DEBUG] Detected {total_filler_words} filler words")

    print("⏸️ [DEBUG] Detecting significant pauses...")
    pauses = []
    for i in range(1, len(words)):
        pause_duration = words[i]['start'] - words[i-1]['end']
        if pause_duration > 0.5:
            pauses.append({
                "start": words[i-1]['end'],
                "end": words[i]['start'],
                "duration": pause_duration,
                "previous_word": words[i-1]['word'],
                "next_word": words[i]['word']
            })
    print(f"✅ [DEBUG] Detected {len(pauses)} significant pauses")

    grammar_errors = []
    if language_tool:
        print("📝 [DEBUG] Checking grammar...")
        sentences = sent_tokenize(transcript_text)
        for sentence in sentences:
            matches = language_tool.check(sentence)
            for match in matches:
                grammar_errors.append({
                    "error": match.message,
                    "context": match.context,
                    "suggestions": match.replacements[:3] if match.replacements else []
                })
        print(f"⚠️ [DEBUG] Found {len(grammar_errors)} grammar errors")

    print("📊 [DEBUG] Compiling analysis data...")
    analysis = {
        "duration_seconds": round(total_duration, 2),
        "word_count": word_count,
        "words_per_minute": wpm,
        "filler_words": {
            "total_count": total_filler_words,
            "percentage": round((total_filler_words / word_count) * 100, 2) if word_count > 0 else 0,
            "breakdown": filler_word_counts
        },
        "pauses": {
            "count": len(pauses),
            "total_duration": round(sum(p['duration'] for p in pauses), 2),
            "average_duration": round(sum(p['duration'] for p in pauses) / len(pauses), 2) if pauses else 0,
            "details": pauses
        },
        "grammar_errors": {
            "count": len(grammar_errors),
            "details": grammar_errors
        }
    }
    print("💾 [DEBUG] Saving analysis...")
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"✅ [DEBUG] Analysis saved: {analysis_path}")


@app.route("/video")
def serve_video():
    print("🎬 [DEBUG] Serving video endpoint called")
    print("🔍 [DEBUG] Looking for latest video...")
    list_of_videos = glob.glob(os.path.join(UPLOAD_FOLDER, "recording_*.mp4"))
    if not list_of_videos:
        print("❌ [DEBUG] No video found")
        return "No video found", 404
    latest_video = max(list_of_videos, key=os.path.getctime)
    print(f"✅ [DEBUG] Serving video: {latest_video}")
    return send_from_directory(UPLOAD_FOLDER, os.path.basename(latest_video))


@app.route("/report")
def serve_report():
    print("📊 [DEBUG] Serving posture report endpoint called")
    print("🔍 [DEBUG] Looking for latest posture report...")
    list_of_reports = glob.glob(os.path.join(
        POSTURE_FOLDER, "posture_report_*.json"))
    if not list_of_reports:
        print("❌ [DEBUG] No posture report found")
        return jsonify({"error": "No report found"}), 404
    latest_report = max(list_of_reports, key=os.path.getctime)
    print(f"✅ [DEBUG] Serving report: {latest_report}")
    with open(latest_report, 'r') as f:
        return jsonify(json.load(f))


@app.route("/combined_report")
def serve_combined_report():
    print("📦 [DEBUG] Serving combined report endpoint called")
    print("🔍 [DEBUG] Looking for latest combined report...")
    list_of_reports = glob.glob(os.path.join(REPORTS_FOLDER, "report_*.json"))
    if not list_of_reports:
        print("❌ [DEBUG] No combined report found")
        return jsonify({"error": "No combined report found"}), 404
    latest_report = max(list_of_reports, key=os.path.getctime)
    print(f"✅ [DEBUG] Serving combined report: {latest_report}")
    with open(latest_report, 'r') as f:
        return jsonify(json.load(f))


@app.route("/transcript")
def serve_transcript():
    print("📝 [DEBUG] Serving transcript endpoint called")
    print("🔍 [DEBUG] Looking for latest transcript...")
    list_of_transcripts = glob.glob(os.path.join(
        TRANSCRIPTS_FOLDER, "transcript_*.json"))
    if not list_of_transcripts:
        print("❌ [DEBUG] No transcript found")
        return jsonify({"error": "No transcript found"}), 404
    latest_transcript = max(list_of_transcripts, key=os.path.getctime)
    print(f"✅ [DEBUG] Serving transcript: {latest_transcript}")
    with open(latest_transcript, 'r') as f:
        return jsonify(json.load(f))


@app.route("/analysis")
def serve_analysis():
    print("🔍 [DEBUG] Serving analysis endpoint called")
    print("🔍 [DEBUG] Looking for latest analysis...")
    list_of_analyses = glob.glob(os.path.join(
        ANALYSIS_FOLDER, "analysis_*.json"))
    if not list_of_analyses:
        print("❌ [DEBUG] No analysis found")
        return jsonify({"error": "No analysis found"}), 404
    latest_analysis = max(list_of_analyses, key=os.path.getctime)
    print(f"✅ [DEBUG] Serving analysis: {latest_analysis}")
    with open(latest_analysis, 'r') as f:
        return jsonify(json.load(f))


def get_latest_json_file():
    print("🔍 [DEBUG] Looking for latest report file...")
    json_files = glob.glob(os.path.join(REPORTS_FOLDER, "report_*.json"))
    if json_files:
        latest_file = max(json_files, key=os.path.getctime)
        print(f"✅ [DEBUG] Latest report file found: {latest_file}")
        return latest_file
    print("❌ [DEBUG] No report files found")
    return None


def query_gemini(prompt):
    print("🤖 [DEBUG] Querying Gemini API...")
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        print("✅ [DEBUG] Gemini API response received")
        return response.text.strip() if response.text else "Error processing request."
    except Exception as e:
        print(f"❌ [ERROR] Gemini API error: {e}")
        return f"Error: {str(e)}"


def save_llm_response(data, timestamp):
    filename = os.path.join(LLM_FOLDER, f"llm_response_{timestamp}.json")
    print(f"💾 [DEBUG] Saving LLM response to {filename}...")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"✅ [DEBUG] LLM response saved: {filename}")


def analyze_report():
    print("🤖 [DEBUG] Starting LLM analysis...")
    latest_file = get_latest_json_file()
    if not latest_file:
        print("❌ [DEBUG] No report found for analysis")
        return {"error": "No report found"}

    print("⏰ [DEBUG] Extracting recording timestamp...")
    match = re.search(r'report_(\d+)\.json', latest_file)
    recording_timestamp = match.group(1) if match else str(int(time.time()))
    print(f"✅ [DEBUG] Recording timestamp: {recording_timestamp}")

    print("📄 [DEBUG] Loading report data...")
    with open(latest_file, 'r', encoding='utf-8') as f:
        report_data = json.load(f)
    print("✅ [DEBUG] Report data loaded")

    print("📝 [DEBUG] Extracting transcript...")
    transcript = report_data.get("transcript", {}).get("results", {}).get(
        "channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "").strip()
    print("✅ [DEBUG] Transcript extracted")

    print("📝 [DEBUG] Generating summary with Gemini...")
    summary_prompt = f"""
    Provide a structured summary in Markdown format:
    ## Summary  
    **Posture Analysis**: Breakdown of posture percentage and confidence.  
    **Speech Transcription**: Duration, word count, filler words, pauses, and grammar errors.  
    **Sentiment Analysis**: Overall sentiment summary.  
    ### Data for Analysis  
    ```json
    {json.dumps(report_data, indent=2)}
    ```
    """
    summary_text = query_gemini(summary_prompt)

    print("😊 [DEBUG] Analyzing sentiment with Gemini...")
    sentiment_prompt = f"Analyze the sentiment of the transcript and return it in Markdown format:\n\n## Sentiment Analysis\n\n{transcript}"
    sentiment_output = query_gemini(sentiment_prompt)

    print("✍️ [DEBUG] Enhancing transcript with Gemini...")
    enhanced_prompt = f"""
    Rewrite the transcript concisely while keeping key points and improving readability.
    Return it in Markdown format.
    ## Enhanced Transcript  
    **Transcript:**  
    {transcript}  
    **Sentiment Analysis:**  
    {sentiment_output}  
    **Summary:**  
    {summary_text}
    """
    enhanced_text = query_gemini(enhanced_prompt)

    print("📊 [DEBUG] Compiling LLM analysis results...")
    result = {
        "summary": summary_text,
        "sentiment": sentiment_output,
        "enhanced_transcript": enhanced_text
    }

    save_llm_response(result, recording_timestamp)

    print("🔄 [DEBUG] Converting Markdown to HTML...")
    html_summary = markdown.markdown(summary_text)
    html_sentiment = markdown.markdown(sentiment_output)
    html_transcript = markdown.markdown(enhanced_text)

    print("✅ [DEBUG] LLM analysis completed")
    return {
        "summary": html_summary,
        "sentiment": html_sentiment,
        "enhanced_transcript": html_transcript
    }


@app.route("/analyze_report", methods=["POST"])
def analyze_report_api():
    print("🚀 [DEBUG] /analyze_report endpoint called")
    return jsonify(analyze_report())


@app.route("/llm_response")
def serve_llm_response():
    print("🚀 [DEBUG] /llm_response endpoint called")
    print("🔍 [DEBUG] Looking for latest report...")
    list_of_reports = glob.glob(os.path.join(REPORTS_FOLDER, "report_*.json"))
    if not list_of_reports:
        print("❌ [DEBUG] No report found")
        return jsonify({"error": "No report found"}), 404
    latest_report = max(list_of_reports, key=os.path.getctime)
    print(f"✅ [DEBUG] Latest report found: {latest_report}")
    match = re.search(r'report_(\d+)\.json', latest_report)
    if not match:
        print("❌ [DEBUG] Invalid report filename")
        return jsonify({"error": "Invalid report filename"}), 500
    recording_timestamp = match.group(1)
    llm_response_file = os.path.join(
        LLM_FOLDER, f"llm_response_{recording_timestamp}.json")
    print(f"🔍 [DEBUG] Checking for LLM response file: {llm_response_file}")
    if not os.path.exists(llm_response_file):
        print(f"❌ [DEBUG] LLM response file not found: {llm_response_file}")
        return jsonify({"error": "No LLM response found"}), 404
    print("📄 [DEBUG] Loading LLM response...")
    with open(llm_response_file, 'r') as f:
        response_data = json.load(f)
    print(f"✅ [DEBUG] Serving LLM response: {llm_response_file}")
    return jsonify({
        "summary": markdown.markdown(response_data['summary']),
        "sentiment": markdown.markdown(response_data['sentiment']),
        "enhanced_transcript": markdown.markdown(response_data['enhanced_transcript'])
    })


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/terms')
def terms():
    return render_template('terms.html')


@app.route('/privacy')
def privacy():
    return render_template('privacy.html')


@app.route('/support')
def support():
    return render_template('support.html')


@app.route('/faqs')
def faqs():
    return render_template('faqs.html')


@app.route('/careers')
def careers():
    return render_template('careers.html')


@app.route("/playback")
def playback():
    print("🎥 [DEBUG] Serving playback page")
    return render_template("playback.html")


@app.route("/practise")
def practise():
    print("🏠 [DEBUG] Serving practise page")
    return render_template("practise.html")


@app.route("/")
def index():
    print("🏠 [DEBUG] Serving index page")
    return render_template("index.html")


if __name__ == "__main__":
    print("🚀 [DEBUG] Starting Flask application...")
    app.run(debug=True,  port=5000)
