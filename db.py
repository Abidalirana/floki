from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, Boolean
from datetime import datetime
from config import DATABASE_URL

# --------------------------
# Base & Engine
# --------------------------
Base = declarative_base()
engine = create_async_engine(DATABASE_URL, echo=False)  # PostgreSQL async engine
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# --------------------------
# Users
# --------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    age = Column(Integer, nullable=True)
    location = Column(String, nullable=True)
    funding_status = Column(String, nullable=True)  # demo, funded, etc.
    account_type = Column(String, nullable=True)    # FTMO, MFF, Apex, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    trades = relationship("Trade", back_populates="user")
    emotions = relationship("Emotion", back_populates="user")
    journals = relationship("Journal", back_populates="user")
    feature_usage = relationship("FeatureUsage", back_populates="user")
    reset_challenges = relationship("ResetChallenge", back_populates="user")
    recovery_plans = relationship("RecoveryPlan", back_populates="user")
    rulebook_votes = relationship("RulebookVote", back_populates="user")
    simulator_logs = relationship("SimulatorLog", back_populates="user")

# --------------------------
# Trades
# --------------------------
class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    instrument = Column(String)
    strategy = Column(String)
    entry_time = Column(DateTime)
    exit_time = Column(DateTime)
    outcome = Column(String)  # win/loss
    r_r_ratio = Column(Float)
    max_drawdown = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="trades")
    emotions = relationship("Emotion", back_populates="trade")

# --------------------------
# Emotions
# --------------------------
class Emotion(Base):
    __tablename__ = "emotions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True)
    emotion_tag = Column(String)        # fear, confidence, anger
    confidence_level = Column(Float)    # 0-100
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="emotions")
    trade = relationship("Trade", back_populates="emotions")

# --------------------------
# Journals
# --------------------------
class Journal(Base):
    __tablename__ = "journals"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    text = Column(Text)
    sentiment_score = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="journals")

# --------------------------
# Feature Usage
# --------------------------
class FeatureUsage(Base):
    __tablename__ = "feature_usage"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    feature_name = Column(String)
    action = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="feature_usage")


# --------------------------
# Reset Challenges
# --------------------------
class ResetChallenge(Base):
    __tablename__ = "reset_challenges"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    challenge_name = Column(String)
    completed = Column(Boolean, default=False)
    progress_percent = Column(Float, default=0.0)
    start_time = Column(DateTime)
    end_time = Column(DateTime)

    user = relationship("User", back_populates="reset_challenges")

# --------------------------
# Recovery Plans
# --------------------------
class RecoveryPlan(Base):
    __tablename__ = "recovery_plans"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    plan_name = Column(String)
    completed = Column(Boolean, default=False)
    progress_percent = Column(Float, default=0.0)
    start_time = Column(DateTime)
    end_time = Column(DateTime)

    user = relationship("User", back_populates="recovery_plans")

# --------------------------
# Rulebook Votes
# --------------------------
class RulebookVote(Base):
    __tablename__ = "rulebook_votes"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    rule_name = Column(String)
    vote = Column(String)  # yes/no/abstain
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="rulebook_votes")

# --------------------------
# Simulator Logs
# --------------------------
class SimulatorLog(Base):
    __tablename__ = "simulator_logs"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="simulator_logs")

# --------------------------
# News Items
# --------------------------
class NewsItem(Base):
    __tablename__ = "news_items"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    url = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    detailed_summary = Column(Text, nullable=True)
    keywords = Column(String, nullable=True)

# --------------------------
# Create All Tables Helper
# --------------------------
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ All tables created successfully!")
