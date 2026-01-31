from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# ===============================
# DATABASE CONFIGURATION
# ===============================

# Use environment variable if available, else fallback
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/health_db"
)

# ===============================
# SQLALCHEMY ENGINE
# ===============================

engine = create_engine(
    DATABASE_URL,
    echo=True,              # Set False in production
    pool_pre_ping=True
)

# ===============================
# SESSION FACTORY
# ===============================

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# ===============================
# BASE MODEL
# ===============================

Base = declarative_base()

# ===============================
# DEPENDENCY (FastAPI)
# ===============================

def get_db():
    """
    FastAPI dependency that provides a database session
    and ensures proper cleanup.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# ===============================
# DATABASE CONFIGURATION
# ===============================

# Use environment variable if available, else fallback
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/health_db"
)

# ===============================
# SQLALCHEMY ENGINE
# ===============================

engine = create_engine(
    DATABASE_URL,
    echo=True,              # Set False in production
    pool_pre_ping=True
)

# ===============================
# SESSION FACTORY
# ===============================

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# ===============================
# BASE MODEL
# ===============================

Base = declarative_base()

# ===============================
# DEPENDENCY (FastAPI)
# ===============================

def get_db():
    """
    FastAPI dependency that provides a database session
    and ensures proper cleanup.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
