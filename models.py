from database import Base
from sqlalchemy import Column, String, Integer, Float, DateTime, JSON
from sqlalchemy.sql import func

class Consumption(Base):
    __tablename__ = "consumption"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, autoincrement=True, primary_key=True)
    type_ = Column(String(20))
    date_ = Column(DateTime(timezone=True))
    period = Column(Integer)
    day = Column(Integer)
    hour = Column(Integer)
    prediction = Column(JSON)
    prediction_time = Column(DateTime(timezone=True), server_default=func.now())
    client_ip = Column(String(20))