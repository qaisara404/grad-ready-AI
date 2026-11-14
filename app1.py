import os
import io
import re
import time
import base64
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from werkzeug.exceptions import HTTPException
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv


# Pustaka Google GenAI & Firebase Admin
from google import genai
from google.genai import types
import firebase_admin
from firebase_admin import credentials, firestore
import PyPDF2


load_dotenv()


base_dir = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(base_dir, 'templates')
app = Flask(__name__, template_folder=template_dir)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


# PENGENDALI RALAT GLOBAL: Pastikan semua laluan /api/ memulangkan JSON dengan status HTTP yang betul
@app.errorhandler(Exception)
def handle_exception(e):
    if request.path.startswith('/api/'):
        code = e.code if isinstance(e, HTTPException) else 500
        return jsonify({'error': str(e), 'status': code}), code
    if isinstance(e, HTTPException):
        return e
    return str(e), 500


# 1. Inisialisasi Firebase Admin SDK (Cloud Firestore)
firebase_key_path = os.path.join(base_dir, 'firebase-key.json')
db = None
if os.path.exists(firebase_key_path):
    try:
        cred = credentials.Certificate(firebase_key_path)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("🔥 Berjaya menyambung ke Cloud Firestore!")
    except Exception as e:
        print(f"⚠️ Ralat Firebase: {str(e)}")


# Penyimpanan pengguna tempatan (Kata laluan dicincang dengan selamat)
LOCAL_USERS = {}


# 2. KUNCI API GEMINI - ENVIRONMENT VARIABLE SAHAJA
api_key = os.getenv("GEMINI_API_KEY", "").strip()


AI_CLIENT = genai.Client(api_key=api_key) if api_key else None


FAST_MODEL_PRIORITY = [
    'gemini-2.5-flash-lite',
    'gemini-2.5-flash',
    'gemini-3.5-flash-lite',
    'gemini-3.5-flash',
    'gemini-3.8-flash',
    'gemini-2.0-flash'
]
ACTIVE_FAST_MODEL = None
CANDIDATE_MODELS = []


def inisialisasi_model_terpantas():
    global CANDIDATE_MODELS
    if CANDIDATE_MODELS or not AI_CLIENT:
        return CANDIDATE_MODELS
    try:
         tersedia = set()
         for m in AI_CLIENT.models.list():
             nama = (getattr(m, 'name', '') or '').replace('models/', '')
             actions = getattr(m, 'supported_actions', []) or []
             if 'generateContent' in actions and 'gemini' in nama.lower():
                 if not any(x in nama for x in ['embedding', 'imagen', 'veo', 'tts', 'audio', 'robotics', 'computer-use']):
                     tersedia.add(nama)


         disusun = []
         for p in FAST_MODEL_PRIORITY:
             if p in tersedia:
                 disusun.append(p)
         for t in sorted(tersedia):
             if 'flash' in t and t not in disusun:
                 disusun.append(t)
         for t in sorted(tersedia):
             if t not in disusun:
                 disusun.append(t)
         if disusun:
             CANDIDATE_MODELS = disusun
             print(f"⚡ Senarai model terpantas sedia digunakan: {CANDIDATE_MODELS[:4]}")
             return CANDIDATE_MODELS
    except Exception as e:
        print(f"⚠️ Guna senarai model statik: {e}")
    CANDIDATE_MODELS = FAST_MODEL_PRIORITY
    return CANDIDATE_MODELS


def arahan_bahasa_ketat(lang):
    if lang == 'EN':
        return (
            "CRITICAL LANGUAGE RULE: You MUST write your ENTIRE response 100% in professional English. "
            "Do NOT use any Malay words, Malay headings, or mixed language under any circumstances. Ensure every table and bullet point is strictly in English."
        )
    return (
        "SYARAT BAHASA WAJIB: Anda MESTI menulis keseluruhan laporan 100% dalam Bahasa Melayu Korporat yang gramatis dan profesional. "
        "Jangan gunakan perenggan atau istilah tajuk dalam Bahasa Inggeris kecuali kata nama khas teknologi."
    )


def panggil_gemini_tulen(contents, lang='MY', max_tokens=1600):
    global ACTIVE_FAST_MODEL
    if not AI_CLIENT:
        msg = "GEMINI_API_KEY is missing or invalid." if lang == 'EN' else "Kunci GEMINI_API_KEY tidak sah."
        return False, msg


    senarai = inisialisasi_model_terpantas()
    if ACTIVE_FAST_MODEL:
        calon = [ACTIVE_FAST_MODEL] + [m for m in senarai if m != ACTIVE_FAST_MODEL]
    else:
        calon = senarai


    last_err = ""
    for model_name in calon[:4]:
        t_mula = time.time()
        try:
            try:
                config = types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=max_tokens,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
                response = AI_CLIENT.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )
            except Exception as cfg_err:
                if "thinking" in str(cfg_err).lower() or "invalid" in str(cfg_err).lower():
                    response = AI_CLIENT.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            temperature=0.3,
                            max_output_tokens=max_tokens
                        )
                    )
                else:
                    raise cfg_err


            if response and response.text and len(response.text.strip()) > 15:
                ACTIVE_FAST_MODEL = model_name
                masa = round(time.time() - t_mula, 2)
                print(f"⚡ Siap dalam {masa}s menggunakan model: {model_name}")
                return True, response.text.strip()
        except Exception as e:
            last_err = str(e)
            print(f"⚠️ Model {model_name} gagal: {last_err[:90]}")
            if ACTIVE_FAST_MODEL == model_name:
                ACTIVE_FAST_MODEL = None
            continue


    msg = (
        f"AI service is busy or quota reached. Please click Try Again. ({last_err[:120]})"
        if lang == 'EN' else
        f"Perkhidmatan AI sedang sibuk atau kuota penuh. Sila klik Cuba Semula. ({last_err[:120]})"
    )
    return False, msg


def ekstrak_skor_dari_teks(teks, default_val=0):
    for baris in teks.split("\n")[:10]:
        match = re.search(r'(?:Skor|Score|Markah|Padanan|Match)\s*[:\-]?\s*(\d{1,3})', baris, re.IGNORECASE)
        if match:
            nilai = int(match.group(1))
            return min(max(nilai, 0), 100)
    return default_val


# ==============================================================================
# API AUTHENTIKASI SELAMAT
# ==============================================================================
@app.route('/api/auth/signup', methods=['POST'])
def api_signup():
    try:
        data = request.get_json(silent=True) or {}
        nama = data.get('nama', '').strip()
        emel = data.get('emel', '').strip().lower()
        katalaluan = data.get('katalaluan', '').strip()
        kursus = data.get('kursus', 'Diploma in Computer Science').strip()
        lang = data.get('lang', 'MY')


        if not nama or not emel or not katalaluan:
            msg = 'Please fill in Full Name, Email, and Password.' if lang == 'EN' else 'Sila lengkapkan Nama Penuh, Emel, dan Kata Laluan.'
            return jsonify({'error': msg}), 400


        if len(katalaluan) < 6:
            msg = 'Password must be at least 6 characters.' if lang == 'EN' else 'Kata laluan mestilah sekurang-kurangnya 6 aksara.'
            return jsonify({'error': msg}), 400


        if emel in LOCAL_USERS:
            msg = 'This email is already registered.' if lang == 'EN' else 'Emel ini telah didaftarkan.'
            return jsonify({'error': msg}), 400


        password_hash = generate_password_hash(katalaluan)
        user_obj = {
            'id': emel,
            'nama_penuh': nama,
            'emel_kptm': emel,
            'password_hash': password_hash,
            'program_pengajian': kursus,
            'tarikh_daftar': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        LOCAL_USERS[emel] = user_obj


        if db:
            try:
                db.collection('users').document(emel).set(user_obj)
            except Exception:
                pass


        return jsonify({
            'success': True,
            'user': {'id': emel, 'nama_penuh': nama, 'program_pengajian': kursus}
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/auth/login', methods=['POST'])
def api_login():
    try:
        data = request.get_json(silent=True) or {}
        emel = data.get('emel', '').strip().lower()
        katalaluan = data.get('katalaluan', '').strip()
        lang = data.get('lang', 'MY')


        if not emel or not katalaluan:
            msg = 'Please enter both email and password.' if lang == 'EN' else 'Sila masukkan emel dan kata laluan.'
            return jsonify({'error': msg}), 400


        udata = LOCAL_USERS.get(emel)
        if not udata and db:
            try:
                user_doc = db.collection('users').document(emel).get()
                if user_doc.exists:
                    udata = user_doc.to_dict()
            except Exception:
                pass


        if not udata:
            msg = 'Account not found. Please sign up first.' if lang == 'EN' else 'Akaun tidak dijumpai. Sila daftar akaun baharu terlebih dahulu.'
            return jsonify({'error': msg}), 404


        stored_hash = udata.get('password_hash', '')
        if not stored_hash or not check_password_hash(stored_hash, katalaluan):
            msg = 'Invalid email or password.' if lang == 'EN' else 'Emel atau kata laluan tidak sah.'
            return jsonify({'error': msg}), 401


        return jsonify({
            'success': True,
            'user': {
                'id': emel,
                'nama_penuh': udata.get('nama_penuh'),
                'program_pengajian': udata.get('program_pengajian', '')
            }
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==============================================================================
# NAVBAR 1: IMBASAN RESUME (ATS) - DWI-BAHASA MUTLAK & GROUNDED AUDIT
# ==============================================================================
@app.route('/api/analyze-resume', methods=['POST'])
def analyze_resume():
    try:
        jawatan = request.form.get('jawatan', '').strip()
        nama_pelajar = request.form.get('nama_pelajar', '').strip()
        lang = request.form.get('lang', 'MY')


        if not jawatan:
            msg = 'Please specify the target job position.' if lang == 'EN' else 'Sila nyatakan sasaran jawatan yang dipohon.'
            return jsonify({'error': msg}), 400


        if 'resume' not in request.files or request.files['resume'].filename == '':
            msg = 'Please upload a PDF resume file.' if lang == 'EN' else 'Sila muat naik fail PDF resume anda.'
            return jsonify({'error': msg}), 400


        file = request.files['resume']
        file_bytes = file.read()
        resume_text = ""


        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes), strict=False)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    resume_text += extracted + "\n"
        except Exception as pdf_err:
            print(f"⚠️ Ralat PyPDF2: {pdf_err}")


        resume_clean = resume_text.strip()
        bil_perkataan = len(resume_clean.split())
        is_empty_or_image = (bil_perkataan < 30)


        nota_ekstraksi = (
            f"EXTRACTED TEXT ({bil_perkataan} words):\n{resume_clean[:12000]}"
            if bil_perkataan >= 20 else
            f"CRITICAL ATS WARNING: Standard text parser (PyPDF2) could only extract {bil_perkataan} words from this PDF! "
            f"This means the PDF is either a flattened image/Canva design or has broken font encoding, making it UNREADABLE by standard ATS machines."
        )


        lang_rule = arahan_bahasa_ketat(lang)


        if lang == 'EN':
            prompt = f"""
            {lang_rule}
            You are an Objective ATS Resume Auditor & Senior Technical Recruiter.
            Analyze the attached PDF resume (both its visual layout and extracted text) strictly against the target role: '{jawatan}'.
            Applicant Name (if logged in): {nama_pelajar or 'Not specified'}


            CRITICAL GROUNDING & INTEGRITY RULES:
            1. Line 1 MUST be: `Score: <integer 0-100>` calculated strictly using this rubric:
               - Role & Qualification Alignment with '{jawatan}' (0-35 pts)
               - Exact Keyword Match in Resume vs '{jawatan}' Requirements (0-25 pts)
               - ATS Formatting & Machine Readability (0-20 pts) -> Deduct heavily if the PDF is colorful, multi-column, uses visual skill bars/icons, or if text extraction failed!
               - Evidence & Impact Quality / STAR Structure (0-20 pts)
            2. DO NOT invent any skills, degrees, or projects that are not in the candidate's actual PDF!
            3. DO NOT fabricate fake metrics or percentages (e.g. do not invent "45% improvement") in STAR rewrites! Use placeholders like `[X]%` or `[number]` where the student should insert their real figure, and base rewrites ONLY on their actual resume experiences.


            Structure your report in clean Markdown:
            # 📊 1. Objective ATS Score & Rubric Breakdown
            Provide a Markdown table showing the exact points awarded for each of the 4 rubric categories above and the total score out of 100.


            ## 🎨 2. ATS Formatting & Visual Readability Audit
            Critique the actual layout of the uploaded PDF (colors, columns, graphics, tables, font readability, and whether standard ATS parsers can read it).


            ## 🎯 3. Role Alignment & Keyword Gap Analysis (For `{jawatan}`)
            * **Actual Candidate Profile Extracted:** Summarize their real education, skills, and experience found in the PDF.
            * **✅ Matching Keywords Found in Resume:** List only keywords actually present in the PDF.
            * **❌ Missing Keywords Required for `{jawatan}`:** List critical keywords for `{jawatan}` that are absent from the resume.


            ## ✨ 4. Grounded STAR Method Rewrites (Based Only on Your Real Experience)
            Take 2-3 actual bullet points or experiences from the candidate's PDF and show how to restructure them using the STAR method (use `[X]%` or `[N]` placeholders for metrics—never invent fake achievements).


            ---
            {nota_ekstraksi}
            """
        else:
            prompt = f"""
            {lang_rule}
            Anda ialah Auditor Sistem ATS & Pakar Pengambilan Pekerja Kanan yang objektif dan tegas.
            Analisis fail PDF resume yang dilampirkan (reka bentuk visual dan teks sebenar) secara khusus berbanding sasaran jawatan: '{jawatan}'.
            Nama Pemohon (jika log masuk): {nama_pelajar or 'Tidak dinyatakan'}


            SYARAT INTEGRITI & GROUNDING MUTLAK:
            1. Baris pertama WAJIB: `Skor: <integer 0-100>` yang dikira secara objektif mengikut rubrik ini:
               - Kesesuaian Kelayakan & Pengalaman dengan '{jawatan}' (0-35 markah)
               - Padanan Kata Kunci Sebenar berbanding Keperluan '{jawatan}' (0-25 markah)
               - Kebolehbacaan Mesin ATS & Format Susun Atur (0-20 markah) -> Tolak markah dengan banyak jika resume berwarna-warni, berlajur kompleks (Canva), guna bar peratusan grafik, atau jika teks gagal diekstrak!
               - Kualiti Bukti Pencapaian & Struktur STAR (0-20 markah)
            2. JANGAN SESEKALI mereka-reka kemahiran, kelulusan, atau projek yang tidak wujud di dalam PDF sebenar pengguna!
            3. JANGAN SESEKALI mencipta angka atau peratusan palsu (contoh: jangan reka "mengurangkan ralat 45%") dalam cadangan STAR! Gunakan tanda letak seperti `[X]%` atau `[jumlah]` untuk diisi sendiri oleh pelajar berdasarkan pengalaman sebenar mereka dalam resume.


            Susun laporan anda menggunakan format Markdown yang kemas:
            # 📊 1. Skor Objektif ATS & Pecahan Rubrik
            Paparkan jadual Markdown yang menunjukkan markah sebenar bagi setiap 4 kategori rubrik di atas beserta jumlah keseluruhan (/100).


            ## 🎨 2. Audit Format Susun Atur & Kebolehbacaan Mesin ATS
            Ulas reka bentuk sebenar PDF yang dimuat naik (penggunaan warna, lajur berganda, grafik/ikon, dan status kebolehbacaan teks oleh perisian ATS).


            ## 🎯 3. Kesesuaian Jawatan & Analisis Jurang Kata Kunci (Sasaran: `{jawatan}`)
            * **Ringkasan Profil Sebenar Pemohon:** Nyatakan pendidikan, pengalaman, dan kemahiran sebenar yang ditemui dalam PDF.
            * **✅ Kata Kunci Sepadan yang Wujud dalam Resume:** Senaraikan kata kunci yang benar-benar ada dalam PDF sahaja.
            * **❌ Kata Kunci Kritikal yang Tiada untuk `{jawatan}`:** Senaraikan kata kunci penting bagi `{jawatan}` yang belum dimasukkan.


            ## ✨ 4. Penambahbaikan Ayat Kaedah STAR (Berasaskan Pengalaman Sebenar Anda Sahaja)
            Ambil 2-3 ayat pengalaman/projek sebenar daripada PDF pemohon dan tunjukkan cara menulisnya semula mengikut struktur STAR dalam jadual perbandingan (gunakan `[X]%` bagi metrik tanpa mereka-reka pencapaian palsu).


            ---
            {nota_ekstraksi}
            """


        def jana_analisis_ats_jujur():
            if is_empty_or_image:
                skor_ats = 35
                if lang == 'EN':
                    return f"""Score: {skor_ats}
### 📊 1. Critical ATS Parsing Audit (Score: {skor_ats}/100)


* **Target Position:** `{jawatan}`


* **⚠️ CRITICAL FAILURE:** The ATS scanner extracted **almost zero text ({bil_perkataan} words)** from your PDF.


* **Why This Happened:** You are likely using an image-heavy, multi-column template (such as highly graphical Canva templates). ATS parsers **cannot read text locked inside images or complex graphical grids**, rendering your resume completely invisible to employers.


* **ATS Compatibility Rating:** **FAILED (Not ATS-Friendly)**


---


### 🔑 2. Formatting Errors Identified (Graphic & Layout Audit)


* **Layout Structure:** Multi-column layout with complex sidebars.


* **Graphic Elements:** Detected graphical rating bars, icons, or color blocks which crash ATS parsing pipelines.


* **Text Extraction Test:** Failed. Text cannot be indexed.


---


### ✨ 3. Urgent Step-by-Step Remedial Actions


1. **Switch to a Single-Column Layout:** Re-write your resume in a standard, clean, single-column Microsoft Word format and export directly to PDF.


2. **Remove All Graphic Elements:** Do not use progress bars for skills, icons next to contact details, or profile photos.


3. **Use Plain Typography:** Stick to standard fonts like Arial, Calibri, or Times New Roman. Ensure headers are plain text."""
                else:
                    return f"""Skor: {skor_ats}
### 📊 1. Audit Kebolehbacaan Pembaca ATS (Skor: {skor_ats}/100)


* **Sasaran Jawatan:** `{jawatan}`


* **⚠️ AMARAN KRITIKAL:** Mesin pengimbas ATS gagal mengekstrak teks **({bil_perkataan} patah perkataan dikesan)** daripada fail PDF anda.


* **Punca Utama Kegagalan:** Anda berkemungkinan besar menggunakan templat resume yang sarat dengan elemen grafik, lajur berkembar, atau bar kemahiran visual (contoh: templat Canva berwarna-warni). Perisian ATS **tidak boleh membaca teks yang tertanam di dalam elemen grafik**, menyebabkan resume anda kosong di mata majikan!


* **Status Keserasian ATS:** **GAGAL (Sangat Tidak Mesra ATS)**


---


### 🔑 2. Analisis Struktur & Reka Bentuk Grafik


* **Susun Atur Dokumen:** Lajur berganda dengan bar sisi berwarna-warni yang mengelirukan algoritma pemetaan.


* **Elemen Visual:** Terdapat ikon, bar kemahiran, atau garisan pembahagi yang memutuskan aliran pembacaan mesin.


* **Ujian Ekstraksi Teks:** Gagal. Teks tersangkut dan tidak boleh diindeks.


---


### ✨ 3. Pelan Tindakan Segera untuk Menyelamatkan Resume Anda


1. **Tukar kepada Format Satu Lajur (Single-Column):** Bina semula resume anda menggunakan Microsoft Word biasa dengan susun atur atas-ke-bawah yang ringkas, kemudian simpan sebagai PDF.


2. **Hapuskan Grafik & Ikon:** Buang semua ikon telefon/emel, gambar profil (kecuali diminta di Malaysia), dan bar peratusan kemahiran.


3. **Gunakan Teks Mudah:** Gunakan fon standard seperti Arial atau Calibri dan pastikan setiap bahagian dipisahkan dengan teks tajuk yang jelas (contoh: `PENDIDIKAN`, `PROJEK`, `PENGALAMAN`)."""


        # OPTIMASI KELAJUAN: Jika teks PDF sudah diekstrak (>=35 patah perkataan), hantar teks terus.
        if bil_perkataan >= 35:
            berjaya, hasil_teks = panggil_gemini_tulen(prompt, lang=lang, max_tokens=1400)
        else:
            pdf_part = types.Part.from_bytes(data=file_bytes, mime_type='application/pdf')
            berjaya, hasil_teks = panggil_gemini_tulen([pdf_part, prompt], lang=lang, max_tokens=1400)


        if not berjaya:
            if bil_perkataan < 20:
                err_pdf = (
                    "Unable to extract readable text from this PDF, and the AI Vision service is currently unreachable. Please upload a text-selectable PDF resume and try again."
                    if lang == 'EN' else
                    "Teks tidak dapat diekstrak daripada fail PDF ini (kemungkinan PDF bergambar/imbasan), dan pelayan AI tidak dapat dihubungi buat sementara waktu. Sila muat naik PDF berasaskan teks atau cuba semula."
                )
                return jsonify({'error': err_pdf}), 422
            return jsonify({'error': hasil_teks}), 503


        skor_ats = ekstrak_skor_dari_teks(hasil_teks, default_val=50)
        return jsonify({'skor_ats': skor_ats, 'laporan': hasil_teks}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==============================================================================
# NAVBAR 2: SIMULASI TEMUDUGA, PENILAIAN & VISION (DWI-BAHASA MUTLAK)
# ==============================================================================
@app.route('/api/generate-questions', methods=['POST'])
def generate_questions():
    try:
        data = request.get_json(silent=True) or {}
        jawatan = data.get('jawatan', '').strip()
        persona = data.get('persona', 'Friendly HR Executive').strip()
        lang = data.get('lang', 'MY')


        if not jawatan:
            msg = 'Please enter the applied position.' if lang == 'EN' else 'Sila masukkan jawatan yang dipohon.'
            return jsonify({'error': msg}), 400


        lang_rule = arahan_bahasa_ketat(lang)
        if lang == 'EN':
            prompt = f"""
            {lang_rule}
            Act as a professional interviewer with persona: '{persona}'.
            Generate 3 realistic, role-specific interview questions for the position of '{jawatan}':
            1. Role Motivation & Background Fit
            2. Technical / Domain-Specific Scenario for '{jawatan}'
            3. Behavioral Question (STAR Method)
            For each question, include a blockquote explaining the 'Interviewer Evaluation Criteria' 100% in English.
            """
        else:
            prompt = f"""
            {lang_rule}
            Bertindak sebagai penemuduga profesional dengan persona: '{persona}'.
            Jana 3 soalan temuduga yang spesifik dan realistik untuk jawatan '{jawatan}':
            1. Motivasi & Kesesuaian Latar Belakang
            2. Senario Penyelesaian Masalah Khusus bagi '{jawatan}'
            3. Soalan Tingkah Laku & Pengurusan Situasi (Kaedah STAR)
            Bagi setiap soalan, sertakan blok petikan (blockquote) yang menerangkan 'Kriteria Penilaian Penemuduga' 100% dalam Bahasa Melayu.
            """


        berjaya, soalan = panggil_gemini_tulen(prompt, lang=lang, max_tokens=900)
        if not berjaya:
            return jsonify({'error': soalan}), 503


        return jsonify({'soalan': soalan}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluate-interview', methods=['POST'])
def evaluate_interview():
    try:
        data = request.get_json(silent=True) or {}
        jawatan = data.get('jawatan', '').strip()
        soalan = data.get('soalan', '').strip()
        jawapan = data.get('jawapan', '').strip()
        wpm = data.get('wpm', 0)
        filler_count = int(data.get('filler_count', 0))
        filler_details = data.get('filler_details', 'None')
        lang = data.get('lang', 'MY')


        if not jawapan:
            msg = 'Please provide an answer to evaluate.' if lang == 'EN' else 'Sila masukkan jawapan anda untuk dinilai.'
            return jsonify({'error': msg}), 400


        lang_rule = arahan_bahasa_ketat(lang)
        if lang == 'EN':
            prompt = f"""
            {lang_rule}
            You are a Strict, Objective Interview Evaluator.
            Evaluate the candidate's exact response below for the role '{jawatan}' 100% in English.
            Interview Question Context: {soalan}
            Candidate's Exact Response: "{jawapan}"
            Supplementary Speech Metrics: {wpm} WPM | Detected Filler Words: {filler_count} ({filler_details}).


            CRITICAL SCORING RUBRIC (Total 100 pts - Do NOT base the score solely on filler words!):
            1. Relevance to the Question & Role (0-25 pts)
            2. Technical / Domain Accuracy & Problem-Solving Depth (0-20 pts)
            3. STAR Structure Completeness (Situation, Task, Action, Result) (0-20 pts)
            4. Clarity & Professional Tone (0-20 pts)
            5. Confidence & Delivery Indicators (WPM & Filler control as supporting metrics) (0-15 pts)


            STRICT INTEGRITY RULE:
            - Line 1 MUST be: `Score: <integer 0-100>`
            - If the candidate's answer is rude/offensive (e.g. "bodo", "stupid"), gibberish, one-word, or completely off-topic, award 0 to 15 points total and firmly explain why it fails corporate interview standards!


            Structure your evaluation in clean Markdown:
            # 🎯 1. Comprehensive Interview Score & Rubric Breakdown
            Show a Markdown table with points awarded for each of the 5 criteria above.


            ## 🔍 2. Honest Critique of Your Answer
            Explain what the candidate did right, what was wrong or missing, and evaluate their tone, clarity, and relevance.


            ## 🗣️ 3. Language & Delivery Refinement
            If informal slang, Manglish, or filler words were used, show a comparison table converting their exact phrases into professional corporate English.


            ## ⭐ 4. Model High-Scoring STAR Answer (For `{jawatan}`)
            Provide a well-structured example response demonstrating how to answer the question effectively.
            """
        else:
            prompt = f"""
            {lang_rule}
            Anda ialah Panel Penilai Temuduga yang Objektif, Tegas, dan Jujur.
            Nilai jawapan sebenar calon di bawah bagi jawatan '{jawatan}' 100% dalam Bahasa Melayu.
            Konteks Soalan Temuduga: {soalan}
            Jawapan Sebenar Calon: "{jawapan}"
            Metrik Pertuturan Tambahan: {wpm} WPM | Kata Pengisi Dikesan: {filler_count} ({filler_details}).


            RUBRIK PEMARKAHAN OBJEKTIF (Jumlah 100 markah - JANGAN kira markah berdasarkan filler words semata-mata!):
            1. Kerelevanan dengan Soalan & Jawatan (0-25 markah)
            2. Ketepatan Isi & Penyelesaian Masalah (0-20 markah)
            3. Kesempurnaan Struktur STAR (Situation, Task, Action, Result) (0-20 markah)
            4. Kejelasan Penyampaian & Adab Profesionalisme (0-20 markah)
            5. Indikator Keyakinan & Kawalan Pertuturan (WPM & Filler sebagai metrik sokongan) (0-15 markah)


            SYARAT KETAT:
            - Baris pertama WAJIB: `Markah: <integer 0-100>`
            - JIKA jawapan calon mengandungi kata kesat/maki (seperti "bodo"), merapu, tidak menjawab soalan, atau hanya sepatah dua kata, berikan markah keseluruhan antara 0 hingga 15 sahaja dan tegur kesilapan tersebut secara tegas!


            Susun penilaian dalam format Markdown yang kemas:
            # 🎯 1. Skor Keseluruhan & Pecahan Rubrik Penilaian
            Paparkan jadual Markdown yang menunjukkan markah bagi setiap 5 kriteria rubrik di atas.


            ## 🔍 2. Analisis Jujur Terhadap Jawapan Anda
            Terangkan apa yang dilakukan dengan betul, apa yang salah atau tertinggal, serta ulas kejelasan dan kerelevanan isi jawapan.


            ## 🗣️ 3. Pembetulan Bahasa Rojak, Loghat & Penyampaian
            Jika terdapat bahasa tidak formal atau kata pengisi dalam jawapan calon, bina jadual perbandingan menukarkan ayat sebenar mereka kepada Bahasa Melayu Korporat.


            ## ⭐ 4. Contoh Jawapan Berstruktur STAR Terbaik (Untuk `{jawatan}`)
            Berikan contoh jawapan lengkap mengikut kaedah STAR sebagai panduan latihan calon.
            """


        berjaya, ulasan = panggil_gemini_tulen(prompt, lang=lang, max_tokens=1200)
        if not berjaya:
            return jsonify({'error': ulasan}), 503


        skor = ekstrak_skor_dari_teks(ulasan, default_val=50)
        return jsonify({'skor': skor, 'ulasan': ulasan}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze-vision', methods=['POST'])
def analyze_vision():
    try:
        data = request.get_json(silent=True) or {}
        image_b64 = data.get('image', '')
        lang = data.get('lang', 'MY')
        if not image_b64:
            msg = 'No camera image received.' if lang == 'EN' else 'Tiada imej kamera diterima.'
            return jsonify({'error': msg}), 400
           
        header, encoded = image_b64.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image_part = types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')
        lang_rule = arahan_bahasa_ketat(lang)


        if lang == 'EN':
            prompt = f"""
            {lang_rule}
            You are a Strict Corporate Image & Interview Readiness Auditor.
            Inspect the attached webcam photo with 100% honesty. Write your ENTIRE report 100% in English.
            - Check Attire: Is the person wearing formal interview attire, casual home t-shirt, or sleepwear/pajamas?
            - Check Grooming & Expression: Is their hair neat or messy/uncombed? Do they look alert or unprepared?
            - Check Camera Framing & Eye Level: Is the camera at eye level or angled poorly? Are they slouching?
            - DO NOT praise the candidate if they are wearing casual/sleepwear, have messy hair, or poor camera alignment! Grade strictly!


            Provide in clean Markdown:
            # 📸 Visual Readiness Score: <0-100>/100
            ## 👔 1. Attire & Grooming Inspection
            ## 👁️ 2. Eye Level, Framing & Facial Expression
            ## 🪑 3. Posture & Background Environment
            ## 🛠️ 4. Immediate Corrections Required Before Interview
            """
        else:
            prompt = f"""
            {lang_rule}
            Anda ialah Auditor Penampilan Korporat & Panel Penemuduga yang tegas dan jujur.
            Periksa gambar sebenar webcam yang dilampirkan ini dengan 100% kejujuran. Tulis laporan 100% dalam Bahasa Melayu.
            - Semak Pakaian: Adakah calon memakai kemeja/pakaian formal temuduga, atau sekadar baju tidur / t-shirt rumah?
            - Semak Kekemasan Diri: Adakah rambut disikat rapi atau serabut/kusut? Adakah wajah kelihatan bersedia?
            - Semak Aras Kamera (Eye Level) & Postur: Adakah kamera terlalu rendah/tinggi atau badan membongkok?
            - JANGAN SESEKALI beri markah tinggi jika calon memakai pakaian tidak formal/baju tidur, rambut serabut, atau postur lemah!


            Susun dalam format Markdown yang kemas:
            # 📸 Skor Kesediaan Visual: <0-100>/100
            ## 👔 1. Audit Pakaian & Kekemasan Diri Sebenar
            ## 👁️ 2. Aras Mata (Eye Level), Bingkai Kamera & Ekspresi Wajah
            ## 🪑 3. Postur Badan & Persekitaran Latar Belakang
            ## 🛠️ 4. Tindakan Pembetulan Segera Sebelum Temuduga
            """


        berjaya, laporan_vision = panggil_gemini_tulen([image_part, prompt], lang=lang, max_tokens=900)
        if not berjaya:
            return jsonify({'error': laporan_vision}), 503


        return jsonify({'laporan_vision': laporan_vision}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==============================================================================
# NAVBAR 3: PADANAN KERJAYA 8 SEKSYEN & SIMULATOR GAJI (DWI-BAHASA MUTLAK)
# ==============================================================================
@app.route('/api/match-job', methods=['POST'])
def match_job():
    try:
        data = request.get_json(silent=True) or {}
        kursus = data.get('kursus', '').strip()
        kemahiran = data.get('kemahiran', '').strip()
        lokasi = data.get('lokasi', '').strip()
        lang = data.get('lang', 'MY')


        if not kursus or not kemahiran:
            msg = 'Please enter your study program and skills.' if lang == 'EN' else 'Sila masukkan program pengajian dan kemahiran anda.'
            return jsonify({'error': msg}), 400


        lang_rule = arahan_bahasa_ketat(lang)
        if lang == 'EN':
            prompt = f"""
            {lang_rule}
            Analyze this student profile strictly: Program: '{kursus}' | Skills: '{kemahiran}' | Location: '{lokasi}'.
            Tailor ALL outputs strictly to '{kursus}'. Distinguish Free vs Paid certifications accurately. Write 100% in English.


            Provide these 8 sections in Markdown:
            # 🧭 1. Career Pathways (5 Options)
            # 🏢 2. Internship Opportunities in {lokasi} (5 Areas)
            # 💼 3. Suitable Job Roles (6-8 Roles: Fit Level, Required Skills & Skill Gap)
            # 🛠️ 4. Technical & Core Skills to Build (8 Skills)
            # 📜 5. Professional Certifications (6-8 Real Certifications: State Paid vs Free & Official Provider)
            # 📚 6. Verified Free Courses (5 Free Courses: MDEC, Google, Microsoft, etc.)
            # 🚀 7. Portfolio Project Ideas (3 Projects for `{kursus}`)
            # 🗺️ 8. Step-by-Step Career Roadmap
            """
        else:
            prompt = f"""
            {lang_rule}
            Analisis profil pelajar ini secara khusus: Program: '{kursus}' | Kemahiran: '{kemahiran}' | Lokasi: '{lokasi}'.
            Semua cadangan MESTILAH mengikut bidang '{kursus}'. Bezakan sijil Percuma vs Berbayar dengan tepat. Tulis 100% dalam Bahasa Melayu.


            Sediakan 8 bahagian ini dalam Markdown:
            # 🧭 1. Laluan Kerjaya Utama (5 Pilihan)
            # 🏢 2. Peluang Latihan Industri di {lokasi} (5 Bidang)
            # 💼 3. Jawatan Bersesuaian (6-8 Jawatan: Tahap Kesesuaian, Kemahiran Diperlukan & Jurang Kemahiran)
            # 🛠️ 4. Kemahiran Teknikal & Teras untuk Dibangunkan (8 Kemahiran)
            # 📜 5. Sijil Profesional Berkaitan (6-8 Sijil Sebenar: Nyatakan Berbayar vs Percuma & Penyedia Rasmi)
            # 📚 6. Kursus Dalam Talian Percuma (5 Kursus Percuma: MDEC Rakyat Digital, Google, Microsoft, dsb.)
            # 🚀 7. Idea Projek Portfolio (3 Projek khusus untuk `{kursus}`)
            # 🗺️ 8. Pelan Pembangunan Kerjaya (Career Roadmap)
            """


        berjaya, laporan = panggil_gemini_tulen(prompt, lang=lang, max_tokens=1600)
        if not berjaya:
            return jsonify({'error': laporan}), 503


        return jsonify({'laporan': laporan}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate-hr-offer', methods=['POST'])
def generate_hr_offer():
    try:
        data = request.get_json(silent=True) or {}
        jenis_rundingan = data.get('jenis_rundingan', '').strip()
        jawatan = data.get('jawatan', '').strip()
        lang = data.get('lang', 'MY')


        if not jawatan:
            msg = 'Please enter the applied role.' if lang == 'EN' else 'Sila masukkan jawatan yang dipohon.'
            return jsonify({'error': msg}), 400


        lang_rule = arahan_bahasa_ketat(lang)
        if lang == 'EN':
            prompt = f"""
            {lang_rule}
            Issue a realistic Malaysian HR initial offer for '{jawatan}' ({jenis_rundingan}). Write 100% in English.
            Line 1 MUST be: `Offer: RM <amount>`
            Line 2 MUST be: `TargetRange: RM <realistic counter range>`
            Then write a concise 2-paragraph official offer letter.
            """
        else:
            prompt = f"""
            {lang_rule}
            Keluarkan surat tawaran awal HR di Malaysia bagi jawatan '{jawatan}' ({jenis_rundingan}). Tulis 100% dalam Bahasa Melayu.
            Baris 1 WAJIB: `Tawaran: RM <jumlah>`
            Baris 2 WAJIB: `Sasaran: RM <julat rundingan munasabah>`
            Kemudian tulis 2 perenggan surat tawaran rasmi yang ringkas dan padat.
            """


        berjaya, respon = panggil_gemini_tulen(prompt, lang=lang, max_tokens=600)
        if not berjaya:
            return jsonify({'error': respon}), 503


        tawaran_dikesan = "RM 2,200"
        sasaran_dikesan = "RM 2,500"
        teks_bersih = []
        for b in respon.split("\n"):
            b_str = b.strip()
            if b_str.lower().startswith("tawaran:") or b_str.lower().startswith("offer:"):
                tawaran_dikesan = re.sub(r'^(?:Tawaran|Offer)\s*:\s*', '', b_str, flags=re.IGNORECASE).strip()
            elif b_str.lower().startswith("sasaran:") or b_str.lower().startswith("targetrange:"):
                sasaran_dikesan = re.sub(r'^(?:Sasaran|TargetRange)\s*:\s*', '', b_str, flags=re.IGNORECASE).strip()
            else:
                teks_bersih.append(b)


        return jsonify({
            'tawaran_ai': tawaran_dikesan,
            'cadangan_sasaran': sasaran_dikesan,
            'surat_tawaran': "\n".join(teks_bersih).strip()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/negotiate-salary', methods=['POST'])
def negotiate_salary():
    try:
        data = request.get_json(silent=True) or {}
        jenis_rundingan = data.get('jenis_rundingan', '').strip()
        jawatan = data.get('jawatan', '').strip()
        tawaran_semasa = data.get('tawaran_semasa', '').strip()
        sasaran_pelajar = data.get('sasaran_pelajar', '').strip()
        hujah_pelajar = data.get('hujah_pelajar', '').strip()
        lang = data.get('lang', 'MY')


        if not hujah_pelajar:
            msg = 'Please enter your negotiation pitch.' if lang == 'EN' else 'Sila masukkan ayat rundingan anda.'
            return jsonify({'error': msg}), 400


        lang_rule = arahan_bahasa_ketat(lang)
        if lang == 'EN':
            prompt = f"""
            {lang_rule}
            Evaluate this salary negotiation pitch for '{jawatan}' ({jenis_rundingan}) 100% in English.
            HR Offer: {tawaran_semasa} | Candidate Target: {sasaran_pelajar} | Pitch: "{hujah_pelajar}".
            # ⚖️ 1. Strategy Score (/100) & Honest Critique
            ## 💼 2. HR Manager Roleplay Decision
            ## ✨ 3. Golden Counter-Offer Script
            """
        else:
            prompt = f"""
            {lang_rule}
            Nilai ayat rundingan gaji ini bagi jawatan '{jawatan}' ({jenis_rundingan}) 100% dalam Bahasa Melayu.
            Tawaran HR: {tawaran_semasa} | Sasaran Calon: {sasaran_pelajar} | Ayat Calon: "{hujah_pelajar}".
            # ⚖️ 1. Skor Strategi (/100) & Kritikan Jujur
            ## 💼 2. Keputusan Simulasi Pengurus HR
            ## ✨ 3. Skrip Rundingan Emas Terbaik
            """


        berjaya, hasil = panggil_gemini_tulen(prompt, lang=lang, max_tokens=1000)
        if not berjaya:
            return jsonify({'error': hasil}), 503


        return jsonify({'hasil': hasil}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==============================================================================
# NAVBAR 4: JOB MATCH (6-STEP WORKFLOW - DWI-BAHASA MUTLAK)
# ==============================================================================
@app.route('/api/job-tailor', methods=['POST'])
def job_tailor():
    try:
        teks_iklan = request.form.get('teks_iklan', '').strip()
        profil_pelajar = request.form.get('profil_pelajar', '').strip()
        nama_pelajar = request.form.get('nama_pelajar', '').strip()
        lang = request.form.get('lang', 'MY')
        imbas_imej = request.files.get('imej_iklan')


        ada_imej = imbas_imej and imbas_imej.filename != ''
        if not ada_imej and not teks_iklan:
            msg = 'Please upload a job ad screenshot or paste the job ad text.' if lang == 'EN' else 'Sila muat naik gambar iklan kerja atau tampal teks iklan.'
            return jsonify({'error': msg}), 400


        if not profil_pelajar:
            msg = 'Please enter your skills and profile summary.' if lang == 'EN' else 'Sila masukkan ringkasan kemahiran dan profil sebenar anda.'
            return jsonify({'error': msg}), 400


        nama_dokumen = nama_pelajar if nama_pelajar else ("[Your Full Name]" if lang == 'EN' else "[Nama Penuh Anda]")
        lang_rule = arahan_bahasa_ketat(lang)


        if lang == 'EN':
            prompt = f"""
            {lang_rule}
            Execute this 6-STEP Job-Resume Matching workflow strictly based on the inputs 100% in English:
            Applicant Name: {nama_dokumen} | Applicant Actual Profile: "{profil_pelajar}" | Job Ad Text: "{teks_iklan}"
            Do NOT claim any experience not present in the Applicant Profile!


            Line 1 MUST be: `Match: <0-100>%`
            # 🔍 STEP 1 & 2: Extracted Job Ad vs Applicant Profile
            ## 🎯 STEP 3, 4 & 5: Compatibility Score, Matching Qualifications & Missing Requirements
            ## 📄 STEP 6A: Tailored Official Cover Letter (Signed by {nama_dokumen})
            ## ✉️ STEP 6B: Ready-to-Send HR Application Email (Signed by {nama_dokumen})
            """
        else:
            prompt = f"""
            {lang_rule}
            Laksanakan ALIRAN KERJA 6-LANGKAH ini secara patuh fakta berdasarkan input 100% dalam Bahasa Melayu:
            Nama Pemohon: {nama_dokumen} | Profil Sebenar Pemohon: "{profil_pelajar}" | Teks Iklan: "{teks_iklan}"
            JANGAN mendakwa pemohon mempunyai pengalaman yang tiada dalam Profil Sebenar!


            Baris 1 WAJIB: `Padanan: <0-100>%`
            # 🔍 LANGKAK 1 & 2: Ekstraksi Iklan Kerja vs Profil Pemohon
            ## 🎯 LANGKAH 3, 4 & 5: Skor Keserasian, Kelayakan Sepadan & Syarat Belum Dipenuhi
            ## 📄 LANGKAH 6A: Surat Iringan Rasmi (Cover Letter Ditandatangani oleh {nama_dokumen})
            ## ✉️ LANGKAH 6B: Draf E-mel Permohonan Kerja kepada HR (Ditandatangani oleh {nama_dokumen})
            """


        if ada_imej:
            image_bytes = imbas_imej.read()
            mime_type = imbas_imej.mimetype or 'image/jpeg'
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            berjaya, hasil = panggil_gemini_tulen([image_part, prompt], lang=lang, max_tokens=1500)
        else:
            berjaya, hasil = panggil_gemini_tulen(prompt, lang=lang, max_tokens=1500)


        if not berjaya:
            return jsonify({'error': hasil}), 503


        skor_num = ekstrak_skor_dari_teks(hasil, default_val=70)
        skor_padanan = f"{skor_num}% Match" if lang == 'EN' else f"{skor_num}% Padan"
        return jsonify({'skor_padanan': skor_padanan, 'hasil': hasil}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==============================================================================
# NAVBAR 5: WORK COACH (DWI-BAHASA MUTLAK)
# ==============================================================================
@app.route('/api/work-coach', methods=['POST'])
def work_coach():
    try:
        data = request.get_json(silent=True) or {}
        senario = data.get('senario', '').strip()
        respon_pelajar = data.get('respon_pelajar', '').strip()
        lang = data.get('lang', 'MY')


        if not respon_pelajar:
            msg = 'Please write your draft response first.' if lang == 'EN' else 'Sila tulis draf respon atau tindakan anda terlebih dahulu.'
            return jsonify({'error': msg}), 400


        lang_rule = arahan_bahasa_ketat(lang)
        if lang == 'EN':
            prompt = f"""
            {lang_rule}
            Evaluate the student's workplace response strictly and honestly 100% in English.
            Scenario: "{senario}" | Student Response: "{respon_pelajar}"


            # 🏆 1. Professionalism & EQ Score: <0-100>/100
            ## ✅ 2. What You Did Right
            ## ❌ 3. What Was Wrong or Inappropriate
            ## 🧠 4. Why That Action / Tone Is Unsuitable
            ## 💬 5. How to Improve & Model Professional Script
            """
        else:
            prompt = f"""
            {lang_rule}
            Nilai respon situasi tempat kerja pelajar ini secara tegas dan jujur 100% dalam Bahasa Melayu.
            Senario: "{senario}" | Respon Pelajar: "{respon_pelajar}"


            # 🏆 1. Skor Profesionalisme & EQ: <0-100>/100
            ## ✅ 2. Apa yang Anda Lakukan dengan Betul
            ## ❌ 3. Apa yang Salah atau Kurang Tepat
            ## 🧠 4. Mengapa Tindakan / Nada Tersebut Kurang Sesuai
            ## 💬 5. Cara Memperbaikinya & Contoh Skrip Profesional Terbaik
            """


        berjaya, hasil = panggil_gemini_tulen(prompt, lang=lang, max_tokens=1100)
        if not berjaya:
            return jsonify({'error': hasil}), 503


        return jsonify({'hasil': hasil}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def home():
    return render_template('index.html')




if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, threaded=True, port=5000)