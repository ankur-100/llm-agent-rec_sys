# db.py
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "Users"
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    user_name = Column(String(100), nullable=False)

class Genre(Base):
    __tablename__ = "Genres"
    genre_id = Column(Integer, primary_key=True, autoincrement=True)
    genre_name = Column(String(50), unique=True, nullable=False)

class UserPreference(Base):
    __tablename__ = "UserPreferences"
    user_id = Column(Integer, ForeignKey("Users.user_id"), primary_key=True)
    genre_id = Column(Integer, ForeignKey("Genres.genre_id"), primary_key=True)
    preference_score = Column(Float, default=0)

class Item(Base):
    __tablename__ = "Items"
    item_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255))
    item_type = Column(String(50))  # 'movie', 'song', 'book'
    genre_id = Column(Integer, ForeignKey("Genres.genre_id"))
    description = Column(Text)

class UserHistory(Base):
    __tablename__ = "UserHistory"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("Users.user_id"))
    item_id = Column(Integer, ForeignKey("Items.item_id"))
    feedback = Column(String(50))  # e.g., 'liked', 'disliked'
    timestamp = Column(DateTime, default=datetime.utcnow)

# Change the connection string as needed (using SQLite for prototyping)
engine = create_engine("sqlite:///recsys.db", echo=False)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(engine)

def get_session():
    return SessionLocal()
