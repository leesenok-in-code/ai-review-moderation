import os
import re
import uuid
import logging
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, redirect, session, flash, jsonify, abort, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dadata_service import get_company_data_by_inn
from dotenv import load_dotenv
import json
import requests
import tensorflow as tf
from sklearn.model_selection import train_test_split

load_model = tf.keras.models.load_model
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
Tokenizer = tf.keras.preprocessing.text.Tokenizer

load_dotenv()

app = Flask(__name__, static_url_path='/static')
app.config['JSON_AS_ASCII'] = False
csrf = CSRFProtect(app)
app.secret_key = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('MYSQL_URI')
app.config['UPLOAD_FOLDER'] = os.path.abspath(os.path.join(app.static_folder, 'uploads'))
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.template_filter('fromjson')
def fromjson_filter(data):
    try:
        return json.loads(data)
    except (TypeError, json.JSONDecodeError):
        return {}
    
upload_folder = os.path.abspath(os.path.join(app.static_folder, 'uploads'))
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder, exist_ok=True)

db = SQLAlchemy(app)

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    person_type = db.Column(db.String(10))
    resident = db.Column(db.String(10))
    inn = db.Column(db.String(20))
    name = db.Column(db.String(255))
    ogrn = db.Column(db.String(255))
    address = db.Column(db.String(255))
    city = db.Column(db.String(255))
    scenario = db.Column(db.String(20))
    seller_name = db.Column(db.String(255))
    email = db.Column(db.String(255), nullable=False)
    seller_city = db.Column(db.String(255))
    seller_phone = db.Column(db.String(20))
    seller_company = db.Column(db.String(255))
    buyer_name = db.Column(db.String(255))
    buyer_city = db.Column(db.String(255))
    buyer_phone = db.Column(db.String(20))
    images = db.Column(db.Text, nullable=True)
    social_links = db.Column(db.Text)
    document_type = db.Column(db.String(20))
    content = db.Column(db.Text)
    filename = db.Column(db.String(255))
    status = db.Column(db.String(20), default='pending')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    admin_comment = db.Column(db.Text)
    seller_is_company = db.Column(db.Boolean, default=False)
    seller_inn = db.Column(db.String(20))
    seller_company_name = db.Column(db.String(255))
    seller_ogrn = db.Column(db.String(255))
    seller_company_address = db.Column(db.String(255))
    seller_company_city = db.Column(db.String(255))
    scenario_type = db.Column(db.String(20), nullable=True)
    amount = db.Column(db.Float, nullable=True)
    currency = db.Column(db.String(10), nullable=True)
    payment_date = db.Column(db.Date, nullable=True)
    planned_payment_date = db.Column(db.Date, nullable=True)
    custom_reason = db.Column(db.Text, nullable=True)
    toxicity_score = db.Column(db.Float)
    toxicity_label = db.Column(db.String(20))

class AdminUser(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True)
    password_hash = db.Column(db.String(512))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

toxicity_model = None
tokenizer = None
MAX_SEQUENCE_LENGTH = 50
MAX_NB_WORDS = 5000
EMBEDDING_DIM = 64

def train_toxicity_model():
    """Обучает модель токсичности если файлы не найдены"""
    app.logger.info("Training toxicity model...")
    
    toxic_phrases = [
        "глупый глупая некрасивый некрасивая урод уродина",
        "умри чмо",
        "ненавижу блять сука дебил тупой лох",
        "сдохни урод выродок хуй нахуй ебаный",
        "глупые тупые вонючие сраные тупые",
        "сдохни сдохните урод выродок хуй нахуй ебаный",
        
    ]

    normal_phrases = [
        "отличный сервис рекомендую всем",
        "спасибо за быструю доставку",
        "качественный товар соответствует описанию",
        "приятные цены хороший ассортимент",
        "вежливый персонал удобный интерфейс",
        "проблем не возникло всё отлично",
        "доволен покупкой буду обращаться снова",
        "оперативная поддержка решили мой вопрос",
        "всё понравилось советую друзьям",
        "быстро качественно надёжно спасибо"
    ]

    data = {
        "text": toxic_phrases + normal_phrases,
        "label": [1]*len(toxic_phrases) + [0]*len(normal_phrases)
    }
    df = pd.DataFrame(data)

    tokenizer = Tokenizer(
        num_words=MAX_NB_WORDS,
        filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
        lower=True
    )
    tokenizer.fit_on_texts(df['text'].values)
    word_index = tokenizer.word_index
    app.logger.info(f"Found {len(word_index)} unique tokens")

    sequences = tokenizer.texts_to_sequences(df['text'].values)
    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    app.logger.info(f"Shape of data tensor: {X.shape}")

    y = np.array(df['label'])
    app.logger.info(f"Shape of label tensor: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    app.logger.info(f"Training data: {X_train.shape}, Test data: {X_test.shape}")

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=MAX_NB_WORDS,
            output_dim=EMBEDDING_DIM,
            input_length=X.shape[1]
        ),
        tf.keras.layers.SpatialDropout1D(0.2),
        tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    model.summary(print_fn=lambda x: app.logger.info(x))

    epochs = 50
    batch_size = 16
    
    app.logger.info("Starting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    app.logger.info(f"Test Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}")

    model.save("toxicity_model.h5")
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    app.logger.info("Toxicity model trained and saved successfully")
    return model, tokenizer

def load_toxicity_model():
    """Загрузка модели для проверки токсичности"""
    global toxicity_model, tokenizer
    
    if toxicity_model is None:
        try:
            if os.path.exists('toxicity_model.h5') and os.path.exists('tokenizer.pickle'):
                app.logger.info("Loading existing toxicity model...")
                toxicity_model = load_model('toxicity_model.h5')
                with open('tokenizer.pickle', 'rb') as handle:
                    tokenizer = pickle.load(handle)
                app.logger.info("Toxicity model loaded successfully")
            else:
                app.logger.warning("Model files not found. Training new model...")
                toxicity_model, tokenizer = train_toxicity_model()
        except Exception as e:
            app.logger.error(f"Error loading toxicity model: {str(e)}")
            toxicity_model = None
            tokenizer = None
    
    return toxicity_model, tokenizer

def check_toxicity(text):
    try:
        if not text.strip():
            return 0.0, 'non-toxic'
        
        model, tokenizer = load_toxicity_model()
        if model is None or tokenizer is None:
            app.logger.error("Model or tokenizer not loaded")
            return 0.0, 'error'
        
        sequence = tokenizer.texts_to_sequences([text])
        
        if not sequence or not sequence[0]:
            return 0.0, 'non-toxic'
        
        padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
        
        prediction = model.predict(padded_sequence, verbose=0)[0][0]
        
        label = 'toxic' if prediction > 0.5 else 'non-toxic'
        return float(prediction), label
        
    except Exception as e:
        app.logger.error(f"Toxicity check error: {str(e)}")
        return 0.0, 'error'

@app.route('/', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        try:
            required_fields = {
                'person_type': 'Тип лица не указан',
                'resident': 'Не указан резидент',
                'scenario': 'Не выбран сценарий',
                'content': 'Текст отзыва обязателен'
            }
            
            seller_is_company = 'seller_is_company' in request.form
            seller_inn = seller_company_name = seller_ogrn = seller_company_address = seller_company_city = None
            
            if seller_is_company:
                seller_inn = request.form.get('seller_inn')
                seller_company_name = request.form.get('seller_company_name')
                seller_ogrn = request.form.get('seller_ogrn')
                seller_company_address = request.form.get('seller_company_address')
                seller_company_city = request.form.get('seller_company_city')

            for field, error_msg in required_fields.items():
                if not request.form.get(field):
                    return jsonify({'success': False, 'error': error_msg}), 400

            email = request.form['email'].strip()
            if not re.fullmatch(r"[^@]+@[^@]+\.[^@]+", email):
                return jsonify({'success': False, 'error': 'Неверный формат email'}), 400

            if request.form['person_type'] == 'jur':
                inn = request.form['inn'].strip()
                if not re.fullmatch(r'\d{10}|\d{12}', inn):
                    return jsonify({'success': False, 'error': 'Некорректный ИНН'}), 400
                
                company_data = get_company_data_by_inn(inn)
                if not company_data:
                    return jsonify({'success': False, 'error': 'Компания не найдена'}), 400
                
                name = company_data['name']
                ogrn = company_data['ogrn']
                address = company_data['address']
                city = company_data['city']
            else:
                inn = name = ogrn = address = city = None

            scenario = request.form['scenario']
            
            amount = None
            currency = None
            payment_date = None
            planned_payment_date = None
            custom_reason = None
            
            if scenario == 'prepayment':
                amount = float(request.form.get('amount', 0))
                currency = request.form.get('currency')
                payment_date_str = request.form.get('payment_date')
                if payment_date_str:
                    payment_date = datetime.strptime(payment_date_str, '%Y-%m-%d').date()
            
            elif scenario == 'noprepayment':
                amount = float(request.form.get('amount2', 0))
                currency = request.form.get('currency2')
                planned_payment_date_str = request.form.get('planned_payment_date')
                if planned_payment_date_str:
                    planned_payment_date = datetime.strptime(planned_payment_date_str, '%Y-%m-%d').date()
            
            elif scenario == 'other':
                custom_reason = request.form.get('custom_reason', '')

            content = request.form['content']
            
            toxicity_score, toxicity_label = check_toxicity(content)
            
            app.logger.info(f"Toxicity check result: score={toxicity_score}, label={toxicity_label}")

            review = Review(
                person_type=request.form['person_type'],
                resident=request.form['resident'],
                inn=inn,
                name=name,
                ogrn=ogrn,
                address=address,
                city=city,
                scenario=request.form['scenario'],
                seller_name=request.form.get('seller_name'),
                seller_city=request.form.get('seller_city'),
                seller_phone=request.form.get('seller_phone'),
                seller_company=request.form.get('seller_company'),
                seller_is_company=seller_is_company,
                seller_inn=seller_inn,
                seller_company_name=seller_company_name,
                seller_ogrn=seller_ogrn,
                seller_company_address=seller_company_address,
                seller_company_city=seller_company_city,
                scenario_type=scenario,
                amount=amount,
                currency=currency,
                payment_date=payment_date,
                planned_payment_date=planned_payment_date,
                custom_reason=custom_reason,
                buyer_name=request.form.get('buyer_name'),
                buyer_city=request.form.get('buyer_city'),
                buyer_phone=request.form.get('buyer_phone'),
                social_links=json.dumps({k: request.form.get(k) for k in [
                    'telegram', 'instagram', 'youtube', 'vk', 'ok', 'fb', 'website'
                ]}),
                document_type=request.form.get('document_type'),
                content=content,
                email=email,
                filename=None,
                toxicity_label=toxicity_label,
                toxicity_score=toxicity_score,
                status='rejected' if toxicity_label == 'toxic' and toxicity_score > 0.7 else 'pending'
            )

            if toxicity_label == 'toxic' and toxicity_score > 0.7:
                review.admin_comment = (f"Автоматическая модерация: обнаружен токсичный контент "
                                       f"(вероятность {toxicity_score*100:.1f}%). "
                                       "Отзыв отклонен автоматически.")

            document_file = request.files.get('document')
            if document_file and document_file.filename:
                if not allowed_file(document_file.filename):
                    return jsonify({'success': False, 'error': 'Недопустимый формат файла для документа'}), 400

                filename = secure_filename(document_file.filename)
                upload_folder = app.config['UPLOAD_FOLDER']
                file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
                unique_filename = f"{uuid.uuid4().hex}.{file_ext}"

                try:
                    document_file.save(os.path.join(upload_folder, unique_filename))
                    review.filename = unique_filename
                except Exception as e:
                    app.logger.error(f"Document save error: {str(e)}")
                    return jsonify({'success': False, 'error': 'Ошибка сохранения документа'}), 500
            else:
                return jsonify({'success': False, 'error': 'Документ обязателен для загрузки'}), 400

            image_files = request.files.getlist('work_images')
            image_filenames = []
            
            for img in image_files:
                if img.filename:
                    if not allowed_file(img.filename):
                        return jsonify({'success': False, 'error': 'Недопустимый формат файла для изображения'}), 400
                    
                    if img.content_length > 2 * 1024 * 1024:
                        return jsonify({'success': False, 'error': 'Размер файла изображения превышает 2MB'}), 400
                    
                    filename = secure_filename(img.filename)
                    unique_filename = f"{uuid.uuid4().hex}.{filename.rsplit('.', 1)[1].lower()}"
                    
                    try:
                        img.save(os.path.join(upload_folder, unique_filename))
                        image_filenames.append(unique_filename)
                    except Exception as e:
                        app.logger.error(f"Image save error: {str(e)}")
                        return jsonify({'success': False, 'error': 'Ошибка сохранения изображения'}), 500
            
            if image_filenames:
                review.images = json.dumps(image_filenames)
                
            db.session.add(review)
            db.session.commit()
            
            if review.status == 'rejected':
                return jsonify({
                    'success': False,
                    'error': 'Отзыв содержит недопустимый контент и был отклонен автоматически',
                    'moderation_comment': review.admin_comment
                }), 400
            else:
                return jsonify({'success': True, 'message': 'Отзыв отправлен на модерацию!'})

        except Exception as e:
            db.session.rollback()
            logging.error(f'Ошибка: {str(e)}')
            return jsonify({'success': False, 'error': 'Внутренняя ошибка сервера'}), 500

    return render_template('form.html')

@app.route('/reviews')
def public_reviews():
    page = request.args.get('page', 1, type=int)
    reviews = Review.query.filter_by(status='approved').paginate(page=page, per_page=10)
    total_reviews = Review.query.filter_by(status='approved').count()
    return render_template('public_reviews.html', reviews=reviews, total=total_reviews)

@app.template_filter('translate_sentiment')
def translate_sentiment_filter(label):
    mapping = {
        'toxic': 'Токсичный',
        'non-toxic': 'Нетоксичный',
        'error': 'Ошибка проверки'
    }
    return mapping.get(label, label)

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        admin = AdminUser.query.filter_by(username=username).first()
        
        if admin and admin.check_password(password):
            session['admin_logged'] = True
            return redirect('/admin/dashboard')
        flash('Неверные учетные данные', 'danger')
    return render_template('admin_login.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin_logged'):
        return redirect('/admin/login')
    
    page = request.args.get('page', 1, type=int)
    reviews = Review.query.filter_by(status='pending').paginate(page=page, per_page=10)
    return render_template('admin_panel.html', reviews=reviews)

@app.route('/admin/approve/<int:id>', methods=['POST'])
def approve_review(id):
    if not session.get('admin_logged'):
        return redirect('/admin/login')
    
    review = Review.query.get_or_404(id)
    review.status = 'approved'
    db.session.commit()
    return redirect('/admin/dashboard')

@app.route('/admin/reject/<int:id>', methods=['POST'])
def reject_review(id):
    if not session.get('admin_logged'):
        return redirect('/admin/login')
    
    review = Review.query.get_or_404(id)
    review.status = 'rejected'
    db.session.commit()
    return redirect('/admin/dashboard')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged', None)
    return redirect('/admin/login')

@app.route('/admin/review/<int:id>')
def admin_review_detail(id):
    if not session.get('admin_logged'):
        return redirect('/admin/login')
    
    review = Review.query.get_or_404(id)
    return render_template('admin_review_detail.html', review=review)

@app.route('/reviews/<int:id>')
def public_review_detail(id):
    review = Review.query.get_or_404(id)
    if review.status != 'approved':
        abort(404)
    return render_template('public_review_detail.html', review=review)

@app.route('/debug/review/<int:id>')
def debug_review(id):
    review = Review.query.get_or_404(id)
    return jsonify({
        'content': review.content,
        'person_type': review.person_type,
        'status': review.status
    })

@app.route('/api/company/<inn>')
def get_company(inn):
    try:
        if not re.match(r'^\d{10,12}$', inn):
            return jsonify({'success': False, 'error': 'Invalid INN format'}), 400
            
        company_data = get_company_data_by_inn(inn)
        
        if not company_data or not isinstance(company_data, dict):
            return jsonify({'success': False, 'error': 'Company not found'}), 404
            
        return jsonify({
            'success': True,
            'data': {
                'name': company_data.get('name', ''),
                'ogrn': company_data.get('ogrn', ''),
                'address': company_data.get('address', ''),
                'city': company_data.get('city', '')
            }
        })
        
    except Exception as e:
        logging.error(f"API handler error: {str(e)}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    
@app.template_filter('translate_scenario')
def translate_scenario(code):
    mapping = {
        'prepayment':     'Предоплата не выполнена',
        'noprepayment':   'Работа выполнена, не оплачено',
        'other':          'Другое'
    }
    return mapping.get(code, code)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('errors/404.html'), 404


if __name__ == '__main__':
    app.run(debug=True)