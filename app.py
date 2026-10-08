from flask import Flask, request, jsonify, send_from_directory
import os
import json 
import re
import io
import requests
import tempfile
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

# --- CLOUDINARY IMPORT ---
import cloudinary
import cloudinary.uploader

# --- ML & OCR IMPORTS ---
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app) 
load_dotenv()

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- ADDED: FORCE SSL FOR TiDB SERVERLESS ---
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'connect_args': {
        'ssl': {
            'ssl_verify_cert': True,
            'ssl_verify_identity': True
        }
    }
}

# --- CLOUDINARY CONFIGURATION ---
cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET'),
    secure=True
)

db = SQLAlchemy(app)


def upload_to_cloudinary(file_obj, folder="tender_app"):
    """Helper function to upload files to Cloudinary and return HTTPS URL."""
    if not file_obj or file_obj.filename == '':
        return None
    try:
        response = cloudinary.uploader.upload(
            file_obj,
            folder=folder,
            resource_type="auto"
        )
        return response.get('secure_url')
    except Exception as e:
        print(f"❌ Cloudinary Upload Error: {str(e)}")
        return None


def get_file_url(path):
    """Helper function to format local paths vs Cloudinary HTTPS URLs."""
    if not path:
        return None
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return f"/uploads/{path}"


def normalize_email(value):
    if value is None:
        return None
    return str(value).strip().lower()


def find_user_by_email(email):
    normalized = normalize_email(email)
    if not normalized:
        return None
    return User.query.filter(db.func.lower(User.email) == normalized).first()


def find_admin_by_email(email):
    normalized = normalize_email(email)
    if not normalized:
        return None
    return Authority.query.filter(db.func.lower(Authority.email) == normalized).first()


# ==========================================
# 1. DATABASE MODELS
# ==========================================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False) 
    company_name = db.Column(db.String(150), nullable=False)
    gstin = db.Column(db.String(50), nullable=False)
    
    job_title = db.Column(db.String(100), nullable=True)
    reg_number = db.Column(db.String(100), nullable=True)
    business_desc = db.Column(db.Text, nullable=True)
    hq_address = db.Column(db.Text, nullable=True)
    pan_number = db.Column(db.String(50), nullable=True)
    
    profile_pic_path = db.Column(db.String(255), nullable=True)
    
    gst_cert_path = db.Column(db.String(255), nullable=False)
    inc_cert_path = db.Column(db.String(255), nullable=False)
    pan_card_path = db.Column(db.String(255), nullable=False)
    
    trade_license_path = db.Column(db.String(255), nullable=True)
    bank_solvency_path = db.Column(db.String(255), nullable=True)
    it_return_path = db.Column(db.String(255), nullable=True)
    iso_certificate_path = db.Column(db.String(255), nullable=True)
    msme_certificate_path = db.Column(db.String(255), nullable=True)
    project_letters_path = db.Column(db.String(255), nullable=True)
    
    status = db.Column(db.String(50), default='pending_verification')

class Authority(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    admin_name = db.Column(db.String(100), nullable=False)
    designation = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    
    dept_name = db.Column(db.String(150), nullable=False)
    office_code = db.Column(db.String(50), nullable=False)
    sector = db.Column(db.String(100), nullable=False)
    hq_address = db.Column(db.Text, nullable=False)
    
    signatory_name = db.Column(db.String(100), nullable=False)
    contact_number = db.Column(db.String(20), nullable=False)
    
    profile_pic_path = db.Column(db.String(255), nullable=True)
    
    auth_letter_path = db.Column(db.String(255), nullable=False)
    id_proof_path = db.Column(db.String(255), nullable=False)
    dsc_key_path = db.Column(db.String(255), nullable=False)
    dept_reg_path = db.Column(db.String(255), nullable=False)
    
    status = db.Column(db.String(50), default='under_review')

class Tender(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    authority_id = db.Column(db.Integer, db.ForeignKey('authority.id'), nullable=True) 

    project_title = db.Column(db.String(255), nullable=False)
    nit_number = db.Column(db.String(100), unique=True, nullable=False)
    project_description = db.Column(db.Text, nullable=False)

    estimated_value = db.Column(db.String(100), nullable=False)
    emd_amount = db.Column(db.String(100), nullable=False)
    sector = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(200), nullable=False)
    deadline = db.Column(db.String(50), nullable=False)

    req_gst_pan = db.Column(db.Boolean, default=True)
    req_affidavit = db.Column(db.Boolean, default=True)
    req_emd_receipt = db.Column(db.Boolean, default=True)
    req_past_work = db.Column(db.Boolean, default=True)
    req_fin_solvency = db.Column(db.Boolean, default=True)
    req_tech_proposal = db.Column(db.Boolean, default=True)
    req_boq = db.Column(db.Boolean, default=True)

    tender_docs = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(50), default='Open')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Bid(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tender_id = db.Column(db.Integer, db.ForeignKey('tender.id'), nullable=False)
    bidder_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    bid_amount = db.Column(db.Float, nullable=False) 
    
    gst_pan_path = db.Column(db.String(255), nullable=True)
    affidavit_path = db.Column(db.String(255), nullable=True)
    emd_receipt_path = db.Column(db.String(255), nullable=True) 
    past_work_path = db.Column(db.String(255), nullable=True) 
    fin_solvency_path = db.Column(db.String(255), nullable=True) 
    technical_doc_path = db.Column(db.String(255), nullable=True)
    financial_doc_path = db.Column(db.String(255), nullable=True)
    
    status = db.Column(db.String(50), default='Pending Review')
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    tech_score = db.Column(db.Float, nullable=True, default=0.0)
    total_score = db.Column(db.Float, nullable=True, default=0.0)

    bidder = db.relationship('User', backref=db.backref('bids', lazy=True))
    tender = db.relationship('Tender', backref=db.backref('bids', lazy=True))


# ==========================================
# 2. MACHINE LEARNING & OCR CORE
# ==========================================

def train_awarding_model():
    X_train = np.array([[100, 100, 100, 100], [0, 0, 0, 100], [100, 100, 100, 80], [50, 50, 50, 90]])
    y_train = np.array([98.0, 10.0, 85.0, 75.0])
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_optimal_bid_predictor():
    X_train = np.array([[10.0], [50.0], [100.0], [250.0], [500.0]])
    y_train = np.array([9.2, 44.5, 88.0, 220.5, 435.0])
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

ml_awarding_model = train_awarding_model()
ml_bid_predictor = train_optimal_bid_predictor()

def extract_text_from_pdf(pdf_path_or_url):
    try:
        text = ""
        # Handle Cloudinary Remote URLs
        if pdf_path_or_url.startswith("http://") or pdf_path_or_url.startswith("https://"):
            response = requests.get(pdf_path_or_url)
            file_bytes = io.BytesIO(response.content)
            reader = PyPDF2.PdfReader(file_bytes)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
            
            if len(text.strip()) < 50:
                print(f"🔍 Scanned Cloudinary document detected. Running Tesseract OCR...")
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_pdf:
                    temp_pdf.write(response.content)
                    temp_pdf.flush()
                    images = convert_from_path(temp_pdf.name)
                    for image in images:
                        text += pytesseract.image_to_string(image) + " "
        else:
            # Handle Local Disk Fallback
            with open(pdf_path_or_url, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted: 
                        text += extracted + " "
            
            if len(text.strip()) < 50:
                print(f"🔍 Scanned local document detected. Activating Tesseract OCR for {os.path.basename(pdf_path_or_url)}...")
                images = convert_from_path(pdf_path_or_url)
                for image in images:
                    text += pytesseract.image_to_string(image) + " "
        return text.lower()
    except Exception as e:
        print(f"⚠️ OCR Error on {pdf_path_or_url}: {str(e)}")
        return ""

def compare_documents(bidder_filepath, authority_tender_text):
    if not bidder_filepath: 
        return 0.0
    
    if bidder_filepath.startswith("http://") or bidder_filepath.startswith("https://"):
        full_path = bidder_filepath
    else:
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], bidder_filepath)
        if not os.path.exists(full_path): 
            return 0.0
    
    bidder_doc_text = extract_text_from_pdf(full_path)
    if len(bidder_doc_text) < 50: 
        return 0.0 

    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([authority_tender_text, bidder_doc_text])
        return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
    except Exception:
        return 0.0

def execute_master_ml_evaluation(tender, bids):
    valid_bids = []
    for bid in bids:
        if all([
            (not tender.req_gst_pan or bid.gst_pan_path),
            (not tender.req_affidavit or bid.affidavit_path),
            (not tender.req_emd_receipt or bid.emd_receipt_path),
            (not tender.req_past_work or bid.past_work_path),
            (not tender.req_fin_solvency or bid.fin_solvency_path),
            (not tender.req_tech_proposal or bid.technical_doc_path),
            (not tender.req_boq or bid.financial_doc_path)
        ]):
            valid_bids.append(bid)
        else:
            bid.status = 'Rejected (Missing Documents)'

    if not valid_bids: 
        return None, "All bids rejected due to missing documents."

    authority_text = f"{tender.project_title} {tender.project_description} {tender.sector} {tender.location}".lower()
    if tender.tender_docs:
        try:
            admin_docs = json.loads(tender.tender_docs)
            for doc in admin_docs:
                doc_path = doc if (doc.startswith("http://") or doc.startswith("https://")) else os.path.join(app.config['UPLOAD_FOLDER'], doc)
                authority_text += " " + extract_text_from_pdf(doc_path)
        except Exception:
            pass

    raw_bids_data = []
    for bid in valid_bids:
        raw_bids_data.append({
            'bid': bid,
            'raw_tech': compare_documents(bid.technical_doc_path, authority_text),
            'raw_past': compare_documents(bid.past_work_path, authority_text),
            'raw_solv': compare_documents(bid.fin_solvency_path, authority_text)
        })

    best_tech = max([data['raw_tech'] for data in raw_bids_data]) or 0.0001
    best_past = max([data['raw_past'] for data in raw_bids_data]) or 0.0001
    best_solv = max([data['raw_solv'] for data in raw_bids_data]) or 0.0001
    l1_price = min(valid_bids, key=lambda x: x.bid_amount).bid_amount

    final_contenders = []
    for data in raw_bids_data:
        bid = data['bid']
        
        relative_tech = (data['raw_tech'] / best_tech) * 100
        relative_past = (data['raw_past'] / best_past) * 100
        relative_solv = (data['raw_solv'] / best_solv) * 100
        f_price_ratio = (l1_price / bid.bid_amount) * 100 if bid.bid_amount > 0 else 0.0

        bid.tech_score = float(relative_tech)
        X_bidder = np.array([[relative_tech, relative_past, relative_solv, f_price_ratio]])
        bid.total_score = float(ml_awarding_model.predict(X_bidder)[0])
        
        if bid.total_score >= 50.0:
            final_contenders.append(bid)
        else:
            bid.status = 'Rejected (Failed ML Quality)'

    if not final_contenders: 
        return None, "No bidders met the minimum AI quality threshold."

    final_contenders.sort(key=lambda x: (-x.total_score, -x.tech_score, x.bid_amount, x.id))
    winner = final_contenders[0]
    
    for bid in valid_bids:
        bid.status = 'Awarded' if bid.id == winner.id else 'Not Awarded'
    
    tender.status = 'Awarded'
    return winner, f"Tender Awarded to {winner.bidder.company_name}"


# ==========================================
# FILE SERVING ENDPOINT
# ==========================================
@app.route('/uploads/<path:filename>', methods=['GET'])
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ==========================================
# 3. BIDDER ENDPOINTS
# ==========================================

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.form
        email = normalize_email(data.get('email'))
        password = (data.get('password') or '').strip()

        if not email or not password:
            return jsonify({"success": False, "message": "Email and password are required"}), 400

        if find_user_by_email(email):
            return jsonify({"success": False, "message": "User already exists"}), 400

        gst = request.files.get('gst_certificate')
        inc = request.files.get('incorporation_certificate')
        pan = request.files.get('pan_card')

        if not all([gst, inc, pan]):
            return jsonify({"success": False, "message": "Missing mandatory documents"}), 400

        gst_url = upload_to_cloudinary(gst, folder="user_docs")
        inc_url = upload_to_cloudinary(inc, folder="user_docs")
        pan_url = upload_to_cloudinary(pan, folder="user_docs")

        new_user = User(
            full_name=data.get('full_name'), email=email,
            password_hash=generate_password_hash(password),
            company_name=data.get('company_name'), gstin=data.get('gstin'),
            gst_cert_path=gst_url, inc_cert_path=inc_url, pan_card_path=pan_url
        )
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"success": True, "message": "Registration successful"}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or request.form
    email = normalize_email(data.get('email'))
    password = data.get('password')

    if not email or not password:
        return jsonify({"success": False, "message": "Missing email or password"}), 400

    user = find_user_by_email(email)

    if user and check_password_hash(user.password_hash, password):
        return jsonify({
            "success": True,
            "message": "Login successful",
            "user": {
                "id": user.id,
                "name": user.full_name,
                "company": user.company_name,
                "status": user.status,
                "fullName": user.full_name,
                "jobTitle": user.job_title,
                "companyName": user.company_name,
                "regNumber": user.reg_number,
                "businessDesc": user.business_desc,
                "hqAddress": user.hq_address,
                "businessEmail": user.email,
                "profileImageUrl": get_file_url(user.profile_pic_path)
            }
        }), 200

    return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route('/get_profile', methods=['GET'])
def get_profile():
    raw_id = request.args.get('user_id') or request.args.get('userId')
    
    if not raw_id:
        return jsonify({"success": False, "message": "Missing user ID."}), 400
        
    user = db.session.get(User, int(raw_id))
    
    if not user: 
        return jsonify({"success": False, "message": "User not found. Session expired."}), 200
    
    return jsonify({
        "success": True,
        "fullName": user.full_name,
        "jobTitle": user.job_title,
        "companyName": user.company_name,
        "regNumber": user.reg_number,
        "businessDesc": user.business_desc,
        "hqAddress": user.hq_address,
        "businessEmail": user.email,
        "profileImageUrl": get_file_url(user.profile_pic_path)
    }), 200

@app.route('/update_profile', methods=['POST'])
def update_profile():
    user_id = request.form.get('user_id')
    user = db.session.get(User, user_id)
    if not user: 
        return jsonify({"success": False, "message": "User not found"}), 200

    try:
        user.full_name = request.form.get('fullName', user.full_name)
        user.job_title = request.form.get('jobTitle', user.job_title)
        user.company_name = request.form.get('companyName', user.company_name)
        user.reg_number = request.form.get('regNumber', user.reg_number)
        user.business_desc = request.form.get('businessDesc', user.business_desc)
        user.hq_address = request.form.get('hqAddress', user.hq_address)
        
        new_email = request.form.get('businessEmail')
        if new_email and new_email != 'null':
            user.email = new_email

        if 'profile_image' in request.files:
            file = request.files['profile_image']
            if file.filename != '':
                pic_url = upload_to_cloudinary(file, folder="profiles")
                if pic_url:
                    user.profile_pic_path = pic_url

        db.session.commit()
        return jsonify({"success": True, "message": "Profile updated successfully!"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/bidder/change-password', methods=['POST'])
def change_bidder_password():
    try:
        data = request.get_json() or {}
        user_id = data.get('userId')
        current_password = data.get('currentPassword')
        new_password = data.get('newPassword')

        if not user_id or not current_password or not new_password:
            return jsonify({"success": False, "message": "Missing required fields."}), 200

        user = db.session.get(User, user_id)
        if not user:
            return jsonify({"success": False, "message": "User not found"}), 200

        if not check_password_hash(user.password_hash, current_password):
            return jsonify({"success": False, "message": "Current password is incorrect"}), 200

        user.password_hash = generate_password_hash(new_password)
        db.session.commit()

        return jsonify({"success": True, "message": "Password updated successfully"}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 200


@app.route('/get_compliance_docs', methods=['GET'])
def get_compliance_docs():
    user_id = request.args.get('user_id')
    user = db.session.get(User, user_id)
    if not user: 
        return jsonify({"success": False, "message": "User not found. Please log out and register again."}), 200
    
    return jsonify({
        "success": True, "status": user.status,
        "documents": {
            "gst": user.gst_cert_path, 
            "pan": user.pan_card_path, 
            "inc": user.inc_cert_path, 
            "trade": user.trade_license_path,
            "bank": user.bank_solvency_path, 
            "it": user.it_return_path, 
            "iso": user.iso_certificate_path,
            "msme": user.msme_certificate_path, 
            "projects": user.project_letters_path
        }
    }), 200

@app.route('/update_compliance_docs', methods=['POST'])
def update_compliance_docs():
    user_id = request.form.get('user_id')
    user = db.session.get(User, user_id)
    if not user: 
        return jsonify({"success": False, "message": "User not found"}), 200

    compliance_field_map = {
        'trade_license': 'trade_license_path',
        'bank_solvency': 'bank_solvency_path',
        'it_return': 'it_return_path',
        'iso_cert': 'iso_certificate_path',
        'msme_cert': 'msme_certificate_path',
        'project_letters': 'project_letters_path'
    }

    for file_key, attr_name in compliance_field_map.items():
        if file_key in request.files:
            file_obj = request.files[file_key]
            if file_obj and file_obj.filename != '':
                doc_url = upload_to_cloudinary(file_obj, folder="compliance_docs")
                if doc_url:
                    setattr(user, attr_name, doc_url)

    db.session.commit()

    check_list = [user.gst_cert_path, user.inc_cert_path, user.pan_card_path, user.trade_license_path, user.bank_solvency_path, user.it_return_path, user.iso_certificate_path, user.project_letters_path]
    if all(check_list):
        user.status = "tender_ready"
        db.session.commit()
        return jsonify({"success": True, "status": "tender_ready", "message": "All docs uploaded!"}), 200

    return jsonify({"success": True, "status": user.status, "message": "Documents updated"}), 200


@app.route('/get_gst_pan', methods=['GET'])
def get_gst_pan():
    user_id = request.args.get('user_id')
    user = db.session.get(User, user_id)
    if not user: 
        return jsonify({"success": False, "message": "User not found. Please log out."}), 200
    return jsonify({"success": True, "gstin": user.gstin, "panNumber": user.pan_number, "gstFileName": user.gst_cert_path, "panFileName": user.pan_card_path}), 200

@app.route('/update_gst_pan', methods=['POST'])
def update_gst_pan():
    user_id = request.form.get('user_id')
    user = db.session.get(User, user_id)
    if not user: 
        return jsonify({"success": False, "message": "User not found"}), 200
    try:
        new_gstin = request.form.get('gstin')
        new_pan_number = request.form.get('pan_number')
        if new_gstin and new_gstin != 'null': user.gstin = new_gstin
        if new_pan_number and new_pan_number != 'null': user.pan_number = new_pan_number

        if 'new_gst_certificate' in request.files:
            gst_file = request.files['new_gst_certificate']
            if gst_file.filename != '':
                gst_url = upload_to_cloudinary(gst_file, folder="user_docs")
                if gst_url:
                    user.gst_cert_path = gst_url

        if 'new_pan_card' in request.files:
            pan_file = request.files['new_pan_card']
            if pan_file.filename != '':
                pan_url = upload_to_cloudinary(pan_file, folder="user_docs")
                if pan_url:
                    user.pan_card_path = pan_url

        db.session.commit()
        return jsonify({"success": True, "message": "GST and PAN updated successfully!"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 200

@app.route('/submit_bid', methods=['POST'])
@app.route('/api/submit-bid', methods=['POST'])
def submit_bid():
    try:
        data = request.form
        tender_id = data.get('tender_id') or data.get('tenderId') or "1"
        bidder_id = data.get('bidder_id') or data.get('bidderId') or "1"
        bid_amount_raw = data.get('bid_amount') or data.get('price')

        if not bid_amount_raw: 
            return jsonify({"success": False, "message": "Missing Price!"}), 400
            
        clean_price = "".join(c for c in str(bid_amount_raw) if c.isdigit() or c == '.')
        if not clean_price: 
            clean_price = "0.0"

        existing_bid = Bid.query.filter_by(tender_id=int(tender_id), bidder_id=int(bidder_id)).first()

        paths = {}
        file_keys = ['technical_doc', 'financial_doc', 'emd_receipt', 'gst_pan_doc', 'affidavit_doc', 'past_work_doc', 'fin_solvency_doc']

        for key in file_keys:
            f = request.files.get(key)
            if f and f.filename != '':
                doc_url = upload_to_cloudinary(f, folder="bids")
                if doc_url:
                    paths[key] = doc_url

        if existing_bid:
            if existing_bid.status == 'Pending Review':
                existing_bid.bid_amount = float(clean_price)
                if 'technical_doc' in paths: existing_bid.technical_doc_path = paths['technical_doc']
                if 'financial_doc' in paths: existing_bid.financial_doc_path = paths['financial_doc']
                if 'emd_receipt' in paths: existing_bid.emd_receipt_path = paths['emd_receipt']
                if 'gst_pan_doc' in paths: existing_bid.gst_pan_path = paths['gst_pan_doc']
                if 'affidavit_doc' in paths: existing_bid.affidavit_path = paths['affidavit_doc']
                if 'past_work_doc' in paths: existing_bid.past_work_path = paths['past_work_doc']
                if 'fin_solvency_doc' in paths: existing_bid.fin_solvency_path = paths['fin_solvency_doc']
                
                existing_bid.submitted_at = datetime.utcnow()
                db.session.commit()
                return jsonify({"success": True, "message": "Bid updated successfully!", "referenceId": f"UPD-BID-{existing_bid.id}"}), 200
            else:
                return jsonify({"success": False, "message": "Cannot edit a bid under evaluation."}), 400
        else:
            new_bid = Bid(
                tender_id=int(tender_id), bidder_id=int(bidder_id), bid_amount=float(clean_price),
                technical_doc_path=paths.get('technical_doc'), financial_doc_path=paths.get('financial_doc'), 
                emd_receipt_path=paths.get('emd_receipt'), gst_pan_path=paths.get('gst_pan_doc'),
                affidavit_path=paths.get('affidavit_doc'), past_work_path=paths.get('past_work_doc'),
                fin_solvency_path=paths.get('fin_solvency_doc'), status='Pending Review'
            )
            db.session.add(new_bid)
            db.session.commit()
            return jsonify({"success": True, "message": "Bid submitted successfully!", "referenceId": f"NEW-BID-{new_bid.id}"}), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/my-bids/<int:user_id>', methods=['GET'])
@app.route('/api/my-bids/', methods=['GET'])
def api_get_my_bids(user_id=None):
    try:
        if user_id is None:
            user_id = request.args.get('user_id') or request.args.get('userId')
        if not user_id:
            return jsonify({"success": False, "message": "Missing user ID."}), 400

        bids = Bid.query.filter_by(bidder_id=int(user_id)).order_by(Bid.submitted_at.desc()).all()
        bid_list = []
        for bid in bids:
            if not bid.tender: 
                continue

            rank, gap_to_l1, l1_price = None, None, None
            if bid.status not in ['Pending Review']:
                all_tender_bids = Bid.query.filter_by(tender_id=bid.tender_id).order_by(Bid.bid_amount.asc()).all()
                if all_tender_bids:
                    l1_bid = all_tender_bids[0]
                    l1_price = f"{l1_bid.bid_amount:,.2f} Cr"
                    for index, tb in enumerate(all_tender_bids):
                        if tb.id == bid.id:
                            rank = f"L{index + 1}"
                            if index > 0:
                                gap = bid.bid_amount - l1_bid.bid_amount
                                gap_to_l1 = f"{gap:,.2f} Cr"
                            break

            bid_list.append({
                "tender_id": str(bid.tender_id), "tender_title": bid.tender.project_title,
                "tender_ref": bid.tender.nit_number, "bid_amount": f"{bid.bid_amount:,.2f} Cr",
                "status": bid.status, "rank": rank, "gapToL1": gap_to_l1, "l1Price": l1_price
            })
        return jsonify({"success": True, "bids": bid_list}), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": "Server error processing bids."}), 500

@app.route('/api/bidder/predict_optimal_bid', methods=['GET'])
def predict_optimal_bid():
    try:
        tender_id = request.args.get('tender_id')
        if not tender_id:
            return jsonify({"success": False, "message": "Missing tender_id"}), 400
            
        tender = db.session.get(Tender, int(tender_id))
        if not tender: 
            return jsonify({"success": False, "message": "Tender not found"}), 404

        clean_est = re.sub(r'[^0-9.]', '', tender.estimated_value)
        estimated_cost_cr = float(clean_est) if clean_est else 0.0
        
        predicted_winning_bid = ml_bid_predictor.predict(np.array([[estimated_cost_cr]]))[0]
        
        return jsonify({
            "success": True,
            "predicted_winning_bid": round(float(predicted_winning_bid), 2),
            "message": f"ML suggests a competitive bid around ₹{predicted_winning_bid:.2f} Cr."
        }), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ==========================================
# 5. AUTHORITY ENDPOINTS
# ==========================================
@app.route('/api/admin/register', methods=['POST'])
def admin_register():
    try:
        data = request.form
        email = normalize_email(data.get('email'))
        password = (data.get('password') or '').strip()

        if not email or not password:
            return jsonify({"success": False, "message": "Email or Password missing"}), 400
        if find_admin_by_email(email):
            return jsonify({"success": False, "message": "Authority email already exists"}), 400

        auth_letter = request.files.get('authLetter')
        id_proof = request.files.get('idProof')
        dsc_key = request.files.get('dscKey')
        dept_reg = request.files.get('deptReg')

        if not all([auth_letter, id_proof, dsc_key, dept_reg]):
            return jsonify({"success": False, "message": "Missing mandatory documents"}), 400

        al_url = upload_to_cloudinary(auth_letter, folder="authority_docs")
        id_url = upload_to_cloudinary(id_proof, folder="authority_docs")
        dsc_url = upload_to_cloudinary(dsc_key, folder="authority_docs")
        dept_url = upload_to_cloudinary(dept_reg, folder="authority_docs")

        new_auth = Authority(
            admin_name=data.get('adminName') or "Unknown", designation=data.get('designation') or "Unknown",
            email=email, password_hash=generate_password_hash(password), dept_name=data.get('deptName') or "Unknown",
            office_code=data.get('officeCode') or "Unknown", sector=data.get('sector') or "Unknown",
            hq_address=data.get('hqAddress') or "Unknown", signatory_name=data.get('signatoryName') or "Unknown",
            contact_number=data.get('contactNumber') or "Unknown", auth_letter_path=al_url, id_proof_path=id_url,
            dsc_key_path=dsc_url, dept_reg_path=dept_url
        )
        db.session.add(new_auth)
        db.session.commit()
        return jsonify({"success": True, "message": "Authority Registration successful"}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": f"Server Error: {str(e)}"}), 500

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    try:
        data = request.get_json(silent=True) or request.form
        email = normalize_email(data.get('email'))
        password = data.get('password')

        if not email or not password:
            return jsonify({"success": False, "message": "Missing info"}), 400

        admin = find_admin_by_email(email)

        if admin and check_password_hash(admin.password_hash, password):
            return jsonify({
                "success": True,
                "message": "Login successful",
                "admin": {
                    "id": admin.id,
                    "adminName": admin.admin_name,
                    "designation": admin.designation,
                    "email": admin.email,
                    "deptName": admin.dept_name,
                    "officeCode": admin.office_code,
                    "sector": admin.sector,
                    "hqAddress": admin.hq_address,
                    "signatoryName": admin.signatory_name,
                    "contactNumber": admin.contact_number,
                    "status": admin.status,
                    "profileImageUrl": get_file_url(admin.profile_pic_path)
                }
            }), 200

        return jsonify({"success": False, "message": "Invalid credentials"}), 401
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/admin/update_profile', methods=['POST'])
def update_authority_profile():
    try:
        admin_id = request.form.get('admin_id')
        
        if not admin_id:
            return jsonify({"success": False, "message": "Admin ID missing"}), 400
            
        admin = db.session.get(Authority, int(admin_id))
        
        if not admin:
            return jsonify({"success": False, "message": "Authority not found"}), 404

        admin.admin_name = request.form.get('adminName', admin.admin_name)
        admin.designation = request.form.get('designation', admin.designation)
        admin.dept_name = request.form.get('deptName', admin.dept_name)
        admin.office_code = request.form.get('officeCode', admin.office_code)
        admin.sector = request.form.get('sector', admin.sector)
        admin.hq_address = request.form.get('hqAddress', admin.hq_address)
        admin.signatory_name = request.form.get('signatoryName', admin.signatory_name)
        admin.contact_number = request.form.get('contactNumber', admin.contact_number)

        if 'profile_image' in request.files:
            file = request.files['profile_image']
            if file.filename != '':
                pic_url = upload_to_cloudinary(file, folder="profiles")
                if pic_url:
                    admin.profile_pic_path = pic_url

        db.session.commit()
        
        new_pic_url = get_file_url(admin.profile_pic_path)
        
        return jsonify({
            "success": True, 
            "message": "Authority Profile updated successfully!",
            "profileImageUrl": new_pic_url
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": f"Server Error: {str(e)}"}), 500

@app.route('/api/admin/change_password', methods=['POST'])
def change_authority_password():
    try:
        data = request.get_json() or {}
        admin_id = data.get('admin_id')
        current_password = data.get('currentPassword')
        new_password = data.get('newPassword')

        if not admin_id or not current_password or not new_password:
            return jsonify({"success": False, "message": "Missing required fields."}), 200

        admin = db.session.get(Authority, int(admin_id))
        if not admin:
            return jsonify({"success": False, "message": "Authority account not found"}), 200

        if not check_password_hash(admin.password_hash, current_password):
            return jsonify({"success": False, "message": "Current password is incorrect"}), 200

        admin.password_hash = generate_password_hash(new_password)
        db.session.commit()

        return jsonify({"success": True, "message": "Password updated successfully!"}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 200

@app.route('/api/admin/get_docs', methods=['GET'])
def get_admin_docs():
    admin_id = request.args.get('admin_id')
    admin = db.session.get(Authority, int(admin_id))
    if not admin: 
        return jsonify({"success": False, "message": "Admin not found."}), 200

    return jsonify({
        "success": True, "status": admin.status,
        "documents": {
            "authLetter": admin.auth_letter_path,
            "idProof": admin.id_proof_path,
            "dscKey": admin.dsc_key_path,
            "deptReg": admin.dept_reg_path
        }
    }), 200

@app.route('/api/admin/update_docs', methods=['POST'])
def update_admin_docs():
    admin_id = request.form.get('admin_id')
    admin = db.session.get(Authority, int(admin_id))
    if not admin: 
        return jsonify({"success": False, "message": "Admin not found."}), 200

    doc_field_map = {
        'authLetter': 'auth_letter_path',
        'idProof': 'id_proof_path',
        'dscKey': 'dsc_key_path',
        'deptReg': 'dept_reg_path'
    }

    for file_key, attr_name in doc_field_map.items():
        if file_key in request.files:
            file_obj = request.files[file_key]
            if file_obj and file_obj.filename != '':
                doc_url = upload_to_cloudinary(file_obj, folder="authority_docs")
                if doc_url:
                    setattr(admin, attr_name, doc_url)

    check_list = [admin.auth_letter_path, admin.id_proof_path, admin.dsc_key_path, admin.dept_reg_path]
    if all(check_list):
        admin.status = "Verified"
    else:
        admin.status = "under_review" 
        
    db.session.commit()

    return jsonify({"success": True, "status": admin.status, "message": "Documents updated successfully!"}), 200

@app.route('/api/admin/create_tender', methods=['POST'], strict_slashes=False)
def create_tender():
    try:
        data = request.form
        uploaded_paths = []
        for file_key in request.files:
            for doc in request.files.getlist(file_key):
                if doc.filename != '':
                    doc_url = upload_to_cloudinary(doc, folder="tenders")
                    if doc_url:
                        uploaded_paths.append(doc_url)
        docs_json = json.dumps(uploaded_paths) if uploaded_paths else "[]"
        
        new_tender = Tender(
            project_title=data.get('projectTitle'), nit_number=data.get('nitNumber'),
            project_description=data.get('projectDescription'), estimated_value=data.get('estimatedValue'),
            emd_amount=data.get('emdAmount'), sector=data.get('sector'),
            location=data.get('location'), deadline=data.get('deadline'),
            
            req_gst_pan=data.get('reqGstPan') == 'true', req_affidavit=data.get('reqAffidavit') == 'true',
            req_emd_receipt=data.get('reqEmdReceipt') == 'true', req_past_work=data.get('reqPastWork') == 'true',
            req_fin_solvency=data.get('reqFinSolvency') == 'true', req_tech_proposal=data.get('reqTechProposal') == 'true',
            req_boq=data.get('reqBoq') == 'true', tender_docs=docs_json, status='Open'
        )
        db.session.add(new_tender)
        db.session.commit()
        return jsonify({"success": True, "message": "Tender Published Successfully!"}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/admin/update_tender', methods=['POST'], strict_slashes=False)
def update_tender():
    try:
        data = request.form
        tender_id = data.get('tender_id')
        if not tender_id: 
            return jsonify({"success": False, "message": "Tender ID missing"}), 400
        tender = db.session.get(Tender, int(tender_id))
        if not tender: 
            return jsonify({"success": False, "message": "Tender not found"}), 404
            
        tender.project_title = data.get('title', tender.project_title)
        tender.estimated_value = data.get('est_value', tender.estimated_value)
        tender.emd_amount = data.get('emd_amount', tender.emd_amount)
        tender.location = data.get('location', tender.location)
        tender.project_description = data.get('description', tender.project_description)
        tender.deadline = data.get('deadline', tender.deadline)
        retained_docs_str = data.get('retained_docs')
        existing_docs = []
        if retained_docs_str:
            try: 
                existing_docs = json.loads(retained_docs_str)
            except Exception: 
                pass
        for file_key in request.files:
            for doc in request.files.getlist(file_key):
                if doc.filename != '':
                    doc_url = upload_to_cloudinary(doc, folder="tenders")
                    if doc_url:
                        existing_docs.append(doc_url)
        tender.tender_docs = json.dumps(existing_docs)
        db.session.commit()
        return jsonify({"success": True, "message": "Tender updated successfully."}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/get_tenders', methods=['GET'], strict_slashes=False)
def get_tenders():
    try:
        tenders = Tender.query.order_by(Tender.created_at.desc()).all()
        tender_list = [{"id": str(t.id), "projectTitle": t.project_title, "nitNumber": t.nit_number, "estimatedValue": t.estimated_value, "deadline": t.deadline, "status": t.status} for t in tenders]
        return jsonify({"success": True, "tenders": tender_list}), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/get_tender/<int:tender_id>', methods=['GET'], strict_slashes=False)
@app.route('/api/get_tender/', methods=['GET'], strict_slashes=False)
def get_single_tender(tender_id=None):
    try:
        if tender_id is None:
            tender_id = request.args.get('tender_id') or request.args.get('id')
        if not tender_id:
            return jsonify({"success": False, "message": "Tender ID missing"}), 400
            
        tender = db.session.get(Tender, int(str(tender_id).strip()))
        if not tender: 
            return jsonify({"success": False, "message": "Tender not found"}), 404
        docs_list = []
        if tender.tender_docs:
            try: 
                docs_list = json.loads(tender.tender_docs)
            except Exception: 
                pass
        return jsonify({
            "success": True,
            "tender": {
                "id": str(tender.id), "projectTitle": tender.project_title, "nitNumber": tender.nit_number,
                "estimatedValue": tender.estimated_value, "emdAmount": tender.emd_amount, "sector": tender.sector,
                "location": tender.location, "deadline": tender.deadline, "projectDescription": tender.project_description,
                "status": tender.status, "tenderDocs": docs_list
            }
        }), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/admin/get_bids/<int:tender_id>', methods=['GET'], strict_slashes=False)
@app.route('/api/admin/get_bids/', methods=['GET'], strict_slashes=False)
def get_bids_for_tender(tender_id=None):
    try:
        if tender_id is None:
            tender_id = request.args.get('tender_id')
        if not tender_id:
            return jsonify({"success": False, "message": "Tender ID missing"}), 400

        bids = Bid.query.filter_by(tender_id=int(tender_id)).order_by(Bid.total_score.desc()).all()
        bid_list = []
        for index, bid in enumerate(bids):
            rank_string = f"L{index + 1}"
            bid_list.append({
                "bidId": str(bid.id), "rank": rank_string, "companyName": bid.bidder.company_name,
                "bidAmount": f"₹{bid.bid_amount:,.2f} Cr", "status": bid.status,
                "isVerified": bid.bidder.status == 'tender_ready',
                "mlScore": round(bid.total_score, 2)
            })
        return jsonify({"success": True, "bids": bid_list}), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/admin/get_bid_details/<int:bid_id>', methods=['GET'], strict_slashes=False)
@app.route('/api/admin/get_bid_details/', methods=['GET'], strict_slashes=False)
def get_bid_details(bid_id=None):
    try:
        if bid_id is None:
            bid_id = request.args.get('bid_id')
        if not bid_id:
            return jsonify({"success": False, "message": "Bid ID missing"}), 400

        bid = db.session.get(Bid, int(bid_id))
        if not bid: 
            return jsonify({"success": False, "message": "Bid not found"}), 404
        return jsonify({
            "success": True,
            "bidDetails": {
                "bidId": str(bid.id), "companyName": bid.bidder.company_name, "tenderEstimate": bid.tender.estimated_value,
                "bidAmount": f"₹{bid.bid_amount:,.2f} Cr", "status": bid.status,
                "techDoc": bid.technical_doc_path, "finDoc": bid.financial_doc_path, "emdDoc": bid.emd_receipt_path,
                "gstDoc": bid.gst_pan_path, "affidavitDoc": bid.affidavit_path, "pastWorkDoc": bid.past_work_path,
                "finSolvencyDoc": bid.fin_solvency_path
            }
        }), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ==========================================
# 6. MANUAL AI TENDER AWARDING ROUTE
# ==========================================
@app.route('/api/admin/auto_award/<int:tender_id>', methods=['POST'], strict_slashes=False)
@app.route('/api/admin/auto_award/', methods=['POST'], strict_slashes=False)
def auto_award_tender(tender_id=None):
    if tender_id is None:
        tender_id = request.form.get('tender_id') or request.args.get('tender_id')
    if not tender_id:
        return jsonify({"success": False, "message": "Tender ID missing"}), 400

    print(f"\n--- [SERVER LOG] Admin Triggered 360° ML Evaluation for Tender {tender_id} ---")
    try:
        tender = db.session.get(Tender, int(tender_id))
        if not tender or tender.status in ['Closed', 'Awarded']:
            return jsonify({"success": False, "message": "Tender unavailable."}), 400

        try:
            deadline_date = None
            for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%m/%d/%Y'):
                try:
                    deadline_date = datetime.strptime(tender.deadline, fmt)
                    break
                except ValueError:
                    pass
            if deadline_date:
                end_of_deadline_day = deadline_date.replace(hour=23, minute=59, second=59)
                if datetime.now() < end_of_deadline_day:
                    return jsonify({"success": False, "message": f"Locked. Wait until {tender.deadline}."}), 403
        except Exception:
            pass

        bids = Bid.query.filter_by(tender_id=int(tender_id)).all()
        if not bids: 
            return jsonify({"success": False, "message": "No bids to evaluate."}), 400

        winner, message = execute_master_ml_evaluation(tender, bids)
        db.session.commit()

        if not winner:
            return jsonify({"success": False, "message": message}), 400

        print(f"🎉 360° ML WINNER DECLARED: {winner.bidder.company_name} with Score: {winner.total_score:.2f}")

        return jsonify({
            "success": True, "message": message, "winner_company": winner.bidder.company_name,
            "score": round(winner.total_score, 2), "tech_score": round(winner.tech_score, 2),
            "winning_amount": f"₹{winner.bid_amount} Cr"
        }), 200

    except Exception as e:
        db.session.rollback()
        print(f"🔥 ML Evaluation Error: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

# ==========================================
# 7. COMPARISON ANALYTICS ENGINE
# ==========================================
@app.route('/api/tender-comparison/<int:tender_id>/<int:user_id>', methods=['GET'], strict_slashes=False)
@app.route('/api/tender-comparison/', methods=['GET'], strict_slashes=False)
def get_tender_comparison(tender_id=None, user_id=None):
    try:
        if tender_id is None:
            tender_id = request.args.get('tender_id')
        if user_id is None:
            user_id = request.args.get('user_id')
            
        if not tender_id or not user_id:
            return jsonify({"success": False, "message": "Missing tender_id or user_id"}), 400

        tender = db.session.get(Tender, int(tender_id))
        all_bids = Bid.query.filter_by(tender_id=int(tender_id)).order_by(Bid.bid_amount.asc()).all()
        
        if not all_bids: 
            return jsonify({"success": False, "message": "No bids found"}), 404

        comparison_list = []
        user_bid_val, user_rank, total_val = 0.0, 0, 0.0

        for idx, b in enumerate(all_bids):
            total_val += b.bid_amount
            is_you = str(b.bidder_id) == str(user_id)
            if is_you:
                user_bid_val = b.bid_amount
                user_rank = idx + 1
            
            comparison_list.append({"companyName": b.bidder.company_name, "amount": b.bid_amount, "rank": f"L{idx + 1}", "isYou": is_you})

        l1_val = all_bids[0].bid_amount
        deviation = ((user_bid_val - l1_val) / l1_val) * 100 if l1_val > 0 else 0

        return jsonify({
            "success": True, "projectTitle": tender.project_title, "nitNumber": tender.nit_number,
            "userBid": user_bid_val, "userRank": f"L{user_rank}", "marketAvg": round(total_val / len(all_bids), 2),
            "deviation": round(deviation, 1), "competitors": len(all_bids), "bids": comparison_list
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ==========================================
# 8. BACKGROUND SCHEDULER & CLOCK FIX
# ==========================================
def background_evaluator():
    with app.app_context():
        open_tenders = Tender.query.filter_by(status='Open').all()
        for tender in open_tenders:
            deadline_date = None
            for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%m/%d/%Y'):
                try:
                    deadline_date = datetime.strptime(tender.deadline, fmt)
                    break
                except ValueError:
                    continue
            
            if deadline_date:
                end_of_day = deadline_date.replace(hour=23, minute=59, second=59)
                if datetime.now() > end_of_day:
                    print(f"\n⏰ [AUTO-CLOCK] Tender {tender.id} deadline passed. Waking up ML Engine...")
                    bids = Bid.query.filter_by(tender_id=tender.id).all()
                    
                    if not bids:
                        print(f"⚠️ No bids for Tender {tender.id}. Closing.")
                        tender.status = 'Closed'
                        db.session.commit()
                        continue
                    
                    winner, msg = execute_master_ml_evaluation(tender, bids)
                    
                    if not winner:
                        print(f"❌ [AUTO-CLOCK] {msg} Closing Tender {tender.id}.")
                        tender.status = 'Closed'
                    else:
                        print(f"🏆 [AUTO-CLOCK] ML Awarded Tender {tender.id} to {winner.bidder.company_name} (Score: {winner.total_score:.2f})")
                    
                    db.session.commit()

scheduler = BackgroundScheduler()
scheduler.add_job(func=background_evaluator, trigger="interval", minutes=1)
scheduler.start()

if __name__ == '__main__':
    with app.app_context():
        db.create_all() 
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)