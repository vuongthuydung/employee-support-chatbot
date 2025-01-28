from sqlalchemy import Column, Integer, String, DateTime
from db.database import Base

class User(Base):
    __tablename__ = "User"

    id = Column(Integer, primary_key=True, index=True)
    user_name = Column(String, unique=True, index=True, nullable=False)
    pass_word = Column(String, nullable=False)
    role = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=True)
