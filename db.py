from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Index, Boolean
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone


def utcnow():
    return datetime.now(timezone.utc)

Base = declarative_base()

class Trade(Base):
    __tablename__ = "trades"
    id        = Column(Integer, primary_key=True)
    symbol    = Column(String, index=True)
    side      = Column(String)
    price     = Column(Float)
    amount    = Column(Float)
    pnl       = Column(Float)
    status    = Column(String)
    reason    = Column(String)
    timestamp = Column(DateTime(timezone=True), default=utcnow, index=True)

class Position(Base):
    __tablename__ = "positions"
    id          = Column(Integer, primary_key=True)
    symbol      = Column(String, unique=True)
    avg_price   = Column(Float)
    amount      = Column(Float)
    side        = Column(String)
    stop_loss   = Column(Float)
    take_profit = Column(Float)
    tp1_hit     = Column(Boolean, default=False)
    trailing_stop_price = Column(Float, nullable=True) 
    updated_at  = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

class SystemState(Base):
    __tablename__ = "system_state"
    key   = Column(String, primary_key=True)
    value = Column(String)

# check_same_thread=False required for async usage
engine  = create_engine("sqlite:///trading.db", connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(engine)

def get_session():
    return Session()
