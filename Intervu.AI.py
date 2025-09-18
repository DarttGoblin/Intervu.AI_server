from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv

from gtts import gTTS
import speech_recognition as sr
import google.generativeai as genai

from pydub import AudioSegment
import tempfile
import json
import os
import io

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
app = Flask(__name__)
CORS(app)

questions_per_duration = {
    "10": {"total": 10, "personal": 2, "technical": 4, "situational": 4},
    "15": {"total": 15, "personal": 3, "technical": 6, "situational": 6},
    "20": {"total": 20, "personal": 4, "technical": 8, "situational": 8}
}

def save_interview_result(result):
    base_dir = os.path.join(os.path.dirname(__file__), "Intervu.AI.Media/interviews")
    os.makedirs(base_dir, exist_ok=True)
    files = [f for f in os.listdir(base_dir) if f.startswith("interview") and f.endswith(".json")]
    numbers = [int(f.replace("interview", "").replace(".json", "")) for f in files]

    if numbers:
        latest_num = max(numbers)
        file_path = os.path.join(base_dir, f"interview{latest_num}.json")
    else:
        latest_num = 1
        file_path = os.path.join(base_dir, "interview1.json")

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
    
    if str(result.get("question_index")) == "1":
        latest_num = (max(numbers) + 1) if numbers else 1
        file_path = os.path.join(base_dir, f"interview{latest_num}.json")
        data = []
    data.append(result)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def reply_to_condidate(question, answer, question_index, condidate_field, condidate_speciality, num_questions):
    # ensure we’re working with string keys
    split = questions_per_duration[str(num_questions)]

    p = split["personal"]
    t = split["technical"]
    s = split["situational"]

    personal_start, personal_end = 1, p
    technical_start, technical_end = personal_end + 1, personal_end + t
    situational_start, situational_end = technical_end + 1, technical_end + s

    prompt = f"""
        You are an AI interview assistant for a virtual interview platform.

        The interview is structured as follows:
        - Questions {personal_start} to {personal_end}: Personal/behavioral questions
        - Questions {technical_start} to {technical_end}: Technical questions specific to the {condidate_speciality} speciality in {condidate_field} field
        - Questions {situational_start} to {situational_end}: Situational/hypothetical questions related to the {condidate_speciality} speciality in {condidate_field} field

        Important:
        - The candidate's response comes from a speech-to-text system, so minor transcription errors or typos may exist.
        - Ignore obvious transcription mistakes that don’t affect meaning.
        - Focus on evaluating the candidate's intended response, not transcription quality.

        Task:
        1. Evaluate the candidate's response to the given question.
        2. Provide constructive feedback:
        - If the response contains real mistakes (not typos), politely correct them in short and concise way.
        - If the response is correct, clarify it to give more depth in short and concise way.
        - In both cases, generate short and concise content, max 25 words.
        - If user response was empty, then probably they have no answer or skipped the question, handle the situation.
        3. Assign a score out of 100 with a clear explanation of the evaluation.
        4. Generate the next interview question based on the current question index: {question_index}, 
           based on the interview structure above, keep the questions simple and easy.

        Input:
        Question: "{question}"
        Response: "{answer.strip()}"

        Return the result strictly in this JSON format:
        {{
            "score": number,
            "explanation": "string",
            "feedback": "string (correction or clarification)",
            "next_question": "string"
        }}
        """

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        result = model.generate_content(contents=prompt, generation_config={"response_mime_type": "application/json"})
        return result.text
    
    except Exception as e:
        return {"error": str(e)}
    


@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return {"error": "No text provided"}, 400

    tts = gTTS(text)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return send_file(buf, mimetype="audio/mpeg")




@app.route("/stt", methods=["POST"])
def stt():
    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_webm:
        audio_file.save(tmp_webm.name)
        webm_path = tmp_webm.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        wav_path = tmp_wav.name
        audio = AudioSegment.from_file(webm_path, format="webm")
        audio.export(wav_path, format="wav")

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
        try: text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError: text = ""

    return jsonify({"text": text})




@app.route("/reply", methods=["POST"])
def reply():
    data = request.get_json()
    question = data.get("question", "")
    answer = data.get("answer", "")
    index = data.get("index", "")
    condidate_field = data.get("condidate_field", "")
    condidate_speciality = data.get("condidate_speciality", "")
    num_questions = data.get("num_questions", "")
    result = reply_to_condidate(question, answer, index, condidate_field, condidate_speciality, num_questions)

    try:
        result_json = json.loads(result)
        score = result_json.get("score")
        explanation = result_json.get("explanation")
        feedback = result_json.get("feedback")
        next_question = result_json.get("next_question")

        save_interview_result({
            "question": question,
            "answer": answer,
            "question_index": index,
            "score": score,
            "explanation": explanation,
            "feedback": feedback,
            "next_question": next_question
        })

        return jsonify({
            "score": score,
            "explanation": explanation,
            "feedback": feedback,
            "next_question": next_question
        })

    except json.JSONDecodeError:
        return jsonify({
            "score": None,
            "explanation": result
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)