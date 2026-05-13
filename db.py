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
    logic_snapshot = Column(String) # JSON field for AI context
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
    from sqlalchemy import event, text
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()
    Base.metadata.create_all(engine)
    
    # Manual Migration for missing columns
    session = Session()
    try:
        # Check and add tp1_hit
        try:
            session.execute(text("SELECT tp1_hit FROM positions LIMIT 1"))
        except Exception:
            session.rollback()
            session.execute(text("ALTER TABLE positions ADD COLUMN tp1_hit BOOLEAN DEFAULT 0"))
            session.commit()
            print("Migration: Added tp1_hit to positions")

        # Check and add trailing_stop_price
        try:
            session.execute(text("SELECT trailing_stop_price FROM positions LIMIT 1"))
        except Exception:
            session.rollback()
            session.execute(text("ALTER TABLE positions ADD COLUMN trailing_stop_price FLOAT"))
            session.commit()
            print("Migration: Added trailing_stop_price to positions")

        # Check and add logic_snapshot to trades
        try:
            session.execute(text("SELECT logic_snapshot FROM trades LIMIT 1"))
        except Exception:
            session.rollback()
            session.execute(text("ALTER TABLE trades ADD COLUMN logic_snapshot TEXT"))
            session.commit()
            print("Migration: Added logic_snapshot to trades")
            
    except Exception as e:
        print(f"Migration warning: {e}")
        session.rollback()
    finally:
        session.close()

def get_session():
    return Session()
