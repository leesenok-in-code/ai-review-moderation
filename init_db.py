from app import app, db, AdminUser, Review
from datetime import datetime

def initialize_database():
    with app.app_context():
        db.create_all()
        
        if not AdminUser.query.first():
            admin = AdminUser(username='admin')
            admin.set_password('admin123') 
            db.session.add(admin)
            print("✅ Администратор создан")
        
        if not Review.query.first():
            sample_review = Review(
                person_type='phys',
                resident='resident',
                scenario='prepayment',
                email='test@example.com',
                content='Пример отзыва для тестирования',
                status='pending',
                created_at=datetime.utcnow()
            )
            db.session.add(sample_review)
            print("✅ Тестовый отзыв создан")
        
        db.session.commit()

if __name__ == '__main__':
    initialize_database()