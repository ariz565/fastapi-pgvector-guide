# FastAPI Interview Questions

## Comprehensive Guide for Web API Development

_Master FastAPI concepts, Flask comparisons, ORMs, and advanced topics for technical interviews_

---

## 📚 Table of Contents

1. [FastAPI Fundamentals](#fastapi-fundamentals)
2. [FastAPI vs Flask Comparison](#fastapi-vs-flask-comparison)
3. [Pydantic & Data Validation](#pydantic--data-validation)
4. [ORM & Database Integration](#orm--database-integration)
5. [Authentication & Security](#authentication--security)
6. [Performance & Async Programming](#performance--async-programming)
7. [Testing & Debugging](#testing--debugging)
8. [Deployment & Production](#deployment--production)
9. [Advanced Topics](#advanced-topics)
10. [Real-world Scenarios](#real-world-scenarios)

---

## FastAPI Fundamentals

### Beginner Level

**Q1: What is FastAPI and what makes it different from other Python web frameworks?**

**Answer**: FastAPI is a modern, high-performance web framework for building APIs with Python 3.7+ based on standard Python type hints. Key differentiators:

- **Automatic API documentation** (OpenAPI/Swagger)
- **Type safety** with Python type hints
- **High performance** (comparable to NodeJS and Go)
- **Built-in data validation** with Pydantic
- **Async support** out of the box
- **Editor support** with autocompletion and error detection

**Q2: How do you create a basic FastAPI application?**

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

**Q3: What are path parameters and query parameters in FastAPI?**

**Answer**:

- **Path parameters**: Variables in the URL path (`/items/{item_id}`)
- **Query parameters**: Parameters after `?` in URL (`/items/?q=value`)

```python
@app.get("/users/{user_id}/items/{item_id}")
async def read_user_item(
    user_id: int,           # Path parameter
    item_id: str,           # Path parameter
    q: str = None,          # Query parameter (optional)
    short: bool = False     # Query parameter with default
):
    return {"user_id": user_id, "item_id": item_id, "q": q, "short": short}
```

**Q4: How does FastAPI handle request body validation?**

**Answer**: FastAPI uses Pydantic models for request body validation:

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = False

@app.post("/items/")
async def create_item(item: Item):
    return item
```

### Intermediate Level

**Q5: Explain FastAPI's dependency injection system.**

**Answer**: Dependencies are reusable functions that can be injected into path operations:

```python
from fastapi import Depends

def common_parameters(q: str = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons

# Database dependency example
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/{user_id}")
async def read_user(user_id: int, db: Session = Depends(get_db)):
    return db.query(User).filter(User.id == user_id).first()
```

**Q6: How do you handle different HTTP methods in FastAPI?**

```python
@app.get("/items/{item_id}")      # GET
@app.post("/items/")              # POST
@app.put("/items/{item_id}")      # PUT
@app.delete("/items/{item_id}")   # DELETE
@app.patch("/items/{item_id}")    # PATCH
@app.head("/items/{item_id}")     # HEAD
@app.options("/items/{item_id}")  # OPTIONS

# Multiple methods on same endpoint
@app.api_route("/items/{item_id}", methods=["GET", "POST"])
async def read_or_create_item(item_id: str, request: Request):
    if request.method == "GET":
        return {"item_id": item_id}
    else:
        return {"created": item_id}
```

**Q7: What are response models and how do you use them?**

**Answer**: Response models define the structure of API responses:

```python
class User(BaseModel):
    id: int
    username: str
    email: str

class UserResponse(BaseModel):
    id: int
    username: str
    # Note: email is excluded for security

@app.get("/users/{user_id}", response_model=UserResponse)
async def read_user(user_id: int):
    user = {"id": user_id, "username": "john", "email": "john@example.com"}
    return user  # FastAPI automatically filters out 'email'
```

### Advanced Level

**Q8: How do you implement custom middleware in FastAPI?**

```python
from fastapi import Request
import time

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Class-based middleware
class LoggingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            print(f"Request: {scope['method']} {scope['path']}")
        await self.app(scope, receive, send)
```

**Q9: Explain FastAPI's background tasks and when to use them.**

```python
from fastapi import BackgroundTasks

def write_log(message: str):
    with open("log.txt", "a") as log:
        log.write(message + "\n")

@app.post("/send-notification/")
async def send_notification(
    email: str,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(write_log, f"Notification sent to {email}")
    return {"message": "Notification sent"}

# For CPU-intensive tasks, use external task queue (Celery, RQ)
```

---

## FastAPI vs Flask Comparison

### Core Differences

**Q10: What are the main architectural differences between FastAPI and Flask?**

| Aspect              | FastAPI                              | Flask                                   |
| ------------------- | ------------------------------------ | --------------------------------------- |
| **Type Hints**      | Built-in, required for full features | Optional, third-party extensions        |
| **Async Support**   | Native async/await                   | Requires additional setup               |
| **Data Validation** | Built-in with Pydantic               | Manual or with extensions (Marshmallow) |
| **Documentation**   | Auto-generated OpenAPI/Swagger       | Manual or with extensions               |
| **Performance**     | Higher (ASGI-based)                  | Lower (WSGI-based)                      |
| **Learning Curve**  | Moderate                             | Easier for beginners                    |

**Q11: Compare request handling in FastAPI vs Flask.**

**FastAPI:**

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    name: str
    age: int

@app.post("/users/")
async def create_user(user: User):  # Automatic validation
    return {"user": user}
```

**Flask:**

```python
from flask import Flask, request, jsonify
from marshmallow import Schema, fields

app = Flask(__name__)

class UserSchema(Schema):
    name = fields.Str(required=True)
    age = fields.Int(required=True)

@app.route("/users/", methods=["POST"])
def create_user():
    schema = UserSchema()
    try:
        user = schema.load(request.json)  # Manual validation
        return jsonify({"user": user})
    except ValidationError as err:
        return jsonify(err.messages), 400
```

### Advantages and Disadvantages

**Q12: What are the advantages of FastAPI over Flask?**

**FastAPI Advantages:**

1. **Automatic API Documentation**: OpenAPI/Swagger UI generated automatically
2. **Type Safety**: Compile-time error detection with type hints
3. **Performance**: ~2-3x faster due to ASGI and async support
4. **Modern Python**: Leverages Python 3.7+ features
5. **Built-in Validation**: Pydantic models provide automatic validation
6. **Editor Support**: Better IDE autocompletion and error detection
7. **Standards Compliance**: Based on OpenAPI and JSON Schema

**Q13: What are the disadvantages of FastAPI compared to Flask?**

**FastAPI Disadvantages:**

1. **Learning Curve**: Requires understanding of async programming and type hints
2. **Newer Ecosystem**: Fewer third-party packages compared to Flask
3. **Python Version**: Requires Python 3.7+
4. **Complexity**: More opinionated, less flexible for simple use cases
5. **Memory Usage**: Slightly higher memory footprint
6. **Debugging**: Async debugging can be more complex

**Q14: When would you choose Flask over FastAPI?**

**Choose Flask when:**

- Building simple web applications with templates
- Working with legacy Python versions (<3.7)
- Team has extensive Flask experience
- Need maximum flexibility and minimal opinions
- Building prototypes quickly
- Extensive use of Flask ecosystem (Flask-Login, Flask-Admin, etc.)

**Choose FastAPI when:**

- Building APIs (especially microservices)
- Need high performance and async support
- Want automatic documentation
- Team comfortable with modern Python
- Need strong type safety
- Building data-heavy applications

### Migration Considerations

**Q15: How would you migrate a Flask application to FastAPI?**

**Migration Strategy:**

1. **Gradual Migration**: Use FastAPI as a proxy to Flask app
2. **Route-by-Route**: Convert endpoints one by one
3. **Data Models**: Convert Flask-SQLAlchemy models to Pydantic
4. **Dependencies**: Replace Flask extensions with FastAPI equivalents
5. **Testing**: Adapt test suite to use FastAPI test client

**Example Migration:**

**Flask (Before):**

```python
from flask import Flask, request, jsonify

@app.route("/api/users/<int:user_id>", methods=["GET"])
def get_user(user_id):
    user = User.query.get_or_404(user_id)
    return jsonify(user.to_dict())
```

**FastAPI (After):**

```python
from fastapi import FastAPI, HTTPException
from .models import User, UserResponse

@app.get("/api/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

---

## Pydantic & Data Validation

### Fundamentals

**Q16: What is Pydantic and how does it integrate with FastAPI?**

**Answer**: Pydantic is a data validation library that uses Python type annotations. In FastAPI, it's used for:

- Request body validation
- Response serialization
- Configuration management
- Data parsing and conversion

```python
from pydantic import BaseModel, validator, Field
from typing import Optional, List
from datetime import datetime

class User(BaseModel):
    id: int
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    age: Optional[int] = Field(None, ge=0, le=120)
    tags: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)

    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.title()

    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "name": "John Doe",
                "email": "john@example.com",
                "age": 30,
                "tags": ["developer", "python"]
            }
        }
```

**Q17: How do you handle nested models and complex data structures?**

```python
class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: str

class Company(BaseModel):
    name: str
    address: Address

class Employee(BaseModel):
    id: int
    name: str
    company: Company
    skills: List[str]
    metadata: Dict[str, Any] = {}

# Usage in FastAPI
@app.post("/employees/")
async def create_employee(employee: Employee):
    return employee
```

**Q18: How do you implement custom validators in Pydantic?**

```python
from pydantic import BaseModel, validator, root_validator
import re

class UserRegistration(BaseModel):
    username: str
    password: str
    confirm_password: str
    email: str

    @validator('username')
    def username_alphanumeric(cls, v):
        assert v.isalnum(), 'Username must be alphanumeric'
        return v

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain uppercase letter')
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain a number')
        return v

    @root_validator
    def passwords_match(cls, values):
        pw1, pw2 = values.get('password'), values.get('confirm_password')
        if pw1 is not None and pw2 is not None and pw1 != pw2:
            raise ValueError('Passwords do not match')
        return values
```

---

## ORM & Database Integration

### SQLAlchemy with FastAPI

**Q19: How do you integrate SQLAlchemy with FastAPI?**

**Database Setup:**

```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "postgresql://user:pass@localhost/db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

**Models:**

```python
# models.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    items = relationship("Item", back_populates="owner")

class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String)
    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="items")
```

**Q20: How do you handle database operations in FastAPI endpoints?**

```python
# schemas.py (Pydantic models)
class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    items: List[Item] = []

    class Config:
        orm_mode = True  # Allows Pydantic to work with SQLAlchemy models

# crud.py
def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = hash_password(user.password)
    db_user = models.User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# main.py
@app.post("/users/", response_model=schemas.User)
async def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db=db, user=user)
```

### Alternative ORMs

**Q21: Compare SQLAlchemy with other ORMs for FastAPI (Tortoise ORM, Databases).**

**SQLAlchemy (Sync/Async):**

```python
# Pros: Mature, feature-rich, large community
# Cons: Complex for simple use cases, sync by default

# Async SQLAlchemy
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

@app.get("/users/{user_id}")
async def get_user(user_id: int, db: AsyncSession = Depends(get_async_db)):
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()
```

**Tortoise ORM:**

```python
# Pros: Async-first, Django-like syntax, FastAPI integration
# Cons: Smaller community, fewer features

from tortoise.models import Model
from tortoise import fields

class User(Model):
    id = fields.IntField(pk=True)
    email = fields.CharField(max_length=50, unique=True)

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return await User.get(id=user_id)
```

**Databases (Raw SQL):**

```python
# Pros: Fast, direct SQL control, async
# Cons: No ORM features, manual query building

import databases

database = databases.Database("postgresql://...")

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    query = "SELECT * FROM users WHERE id = :user_id"
    return await database.fetch_one(query, {"user_id": user_id})
```

**Q22: How do you handle database migrations in FastAPI projects?**

**Using Alembic:**

```bash
# Install and initialize
pip install alembic
alembic init alembic

# Configure alembic.ini and env.py
# Create migration
alembic revision --autogenerate -m "Create users table"

# Apply migration
alembic upgrade head
```

**Migration Script Example:**

```python
# alembic/versions/xxx_create_users_table.py
def upgrade():
    op.create_table('users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(), nullable=True),
        sa.Column('hashed_password', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)

def downgrade():
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
```

---

## Authentication & Security

### JWT Authentication

**Q23: How do you implement JWT authentication in FastAPI?**

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta

# Configuration
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = get_user_by_username(username)
    if user is None:
        raise credentials_exception
    return user

# Protected endpoint
@app.get("/protected")
async def protected_route(current_user: User = Depends(get_current_user)):
    return {"message": f"Hello {current_user.username}"}
```

**Q24: How do you implement OAuth2 with FastAPI?**

```python
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}
```

### Security Best Practices

**Q25: What are the security best practices for FastAPI applications?**

1. **Input Validation**: Always use Pydantic models
2. **Authentication**: Implement proper JWT or OAuth2
3. **HTTPS**: Use TLS in production
4. **CORS**: Configure properly for web apps
5. **Rate Limiting**: Implement to prevent abuse
6. **SQL Injection**: Use parameterized queries
7. **Password Hashing**: Use bcrypt or similar

```python
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/limited")
@limiter.limit("5/minute")
async def limited_endpoint(request: Request):
    return {"message": "This endpoint is rate limited"}
```

---

## Performance & Async Programming

**Q26: Explain async/await in FastAPI and when to use it.**

**Answer**: FastAPI supports both sync and async functions:

```python
# Sync function (runs in thread pool)
@app.get("/sync")
def sync_endpoint():
    time.sleep(1)  # Blocking operation
    return {"message": "sync"}

# Async function (runs in event loop)
@app.get("/async")
async def async_endpoint():
    await asyncio.sleep(1)  # Non-blocking operation
    return {"message": "async"}

# Database operations
@app.get("/users/{user_id}")
async def get_user_async(user_id: int):
    # Use async database client
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/users/{user_id}") as response:
            return await response.json()
```

**When to use async:**

- I/O operations (database, HTTP requests, file operations)
- High concurrency requirements
- WebSocket connections
- Long-running operations that can yield control

**Q27: How do you optimize FastAPI performance?**

1. **Use async for I/O operations**
2. **Connection pooling for databases**
3. **Caching with Redis**
4. **Response compression**
5. **Background tasks for heavy operations**
6. **Proper database indexing**

```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

# Caching
@cache(expire=60)
@app.get("/cached-data")
async def get_cached_data():
    # Expensive operation
    return expensive_computation()

# Connection pooling
from databases import Database
database = Database("postgresql://...", min_size=5, max_size=20)

# Compression
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

---

## Testing & Debugging

**Q28: How do you test FastAPI applications?**

```python
from fastapi.testclient import TestClient
import pytest

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_create_user():
    response = client.post(
        "/users/",
        json={"email": "test@example.com", "password": "secret"}
    )
    assert response.status_code == 200
    assert response.json()["email"] == "test@example.com"

# Async testing
@pytest.mark.asyncio
async def test_async_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/async-endpoint")
    assert response.status_code == 200

# Database testing with fixtures
@pytest.fixture
def db_session():
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()
```

**Q29: How do you debug FastAPI applications?**

1. **Logging**: Use structured logging
2. **Debug mode**: Enable for development
3. **Profiling**: Use tools like py-spy
4. **Error tracking**: Sentry integration

```python
import logging
from fastapi import Request
import time

# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url} - {response.status_code} - {process_time:.4f}s"
    )
    return response

# Error handling
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"message": f"ValueError: {str(exc)}"}
    )
```

---

## Deployment & Production

**Q30: How do you deploy FastAPI applications to production?**

**Docker Deployment:**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Production Server Setup:**

```bash
# Using Gunicorn with Uvicorn workers
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker

# Or with Uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Q31: What are the considerations for FastAPI production deployment?**

1. **ASGI Server**: Use Uvicorn or Hypercorn
2. **Process Manager**: Gunicorn for multiple workers
3. **Reverse Proxy**: Nginx for load balancing
4. **Environment Variables**: Secure configuration
5. **Health Checks**: Implement monitoring endpoints
6. **Logging**: Structured logging with centralized collection
7. **Monitoring**: Prometheus metrics, APM tools

---

## Advanced Topics

**Q32: How do you implement WebSocket connections in FastAPI?**

```python
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(f"Client #{client_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

**Q33: How do you implement custom response classes?**

```python
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
import io
import csv

class CSVResponse(Response):
    media_type = "text/csv"

    def render(self, content) -> bytes:
        output = io.StringIO()
        writer = csv.writer(output)
        for row in content:
            writer.writerow(row)
        return output.getvalue().encode()

@app.get("/export", response_class=CSVResponse)
async def export_data():
    return [["Name", "Age"], ["John", "30"], ["Jane", "25"]]
```

**Q34: How do you handle file uploads and downloads?**

```python
from fastapi import File, UploadFile
import shutil

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    with open(f"uploads/{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "size": file.size}

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"uploads/{filename}"
    return FileResponse(file_path, filename=filename)

# Multiple files
@app.post("/upload-multiple/")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    return [{"filename": f.filename} for f in files]
```

---

## Real-world Scenarios

**Q35: Design a scalable REST API for a social media platform using FastAPI.**

**Answer Structure:**

1. **User Management**: Authentication, profiles, relationships
2. **Content Management**: Posts, comments, likes
3. **Real-time Features**: WebSocket for notifications
4. **Media Handling**: File uploads for images/videos
5. **Caching**: Redis for frequent data
6. **Database**: PostgreSQL with proper indexing
7. **Background Tasks**: Celery for heavy operations
8. **Monitoring**: Health checks and metrics

```python
# Example structure
@app.post("/posts/", response_model=PostResponse)
async def create_post(
    post: PostCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    db_post = crud.create_post(db, post, current_user.id)
    background_tasks.add_task(notify_followers, current_user.id, db_post.id)
    return db_post
```

**Q36: How would you implement a microservices architecture with FastAPI?**

1. **Service Discovery**: Consul or Eureka
2. **API Gateway**: Kong or custom FastAPI gateway
3. **Inter-service Communication**: HTTP or message queues
4. **Shared Authentication**: JWT with shared secret
5. **Database per Service**: Each service owns its data
6. **Event-driven Architecture**: Message brokers for async communication

**Q37: How do you handle API versioning in FastAPI?**

```python
from fastapi import APIRouter

# Version 1
v1_router = APIRouter(prefix="/v1")

@v1_router.get("/users/{user_id}")
async def get_user_v1(user_id: int):
    return {"id": user_id, "name": "John"}

# Version 2
v2_router = APIRouter(prefix="/v2")

@v2_router.get("/users/{user_id}")
async def get_user_v2(user_id: int):
    return {"id": user_id, "name": "John", "created_at": "2024-01-01"}

app.include_router(v1_router)
app.include_router(v2_router)

# Or header-based versioning
@app.get("/users/{user_id}")
async def get_user(user_id: int, request: Request):
    version = request.headers.get("API-Version", "v1")
    if version == "v2":
        return get_user_v2_logic(user_id)
    return get_user_v1_logic(user_id)
```

---

## 🎯 Key Interview Tips

### Technical Preparation

1. **Hands-on Experience**: Build real applications with FastAPI
2. **Understand Async**: Know when and how to use async/await
3. **Database Integration**: Practice with SQLAlchemy and Pydantic
4. **Testing**: Write comprehensive tests for your APIs
5. **Deployment**: Understand production deployment considerations

### Common Pitfalls to Avoid

1. **Mixing sync/async incorrectly**
2. **Not handling exceptions properly**
3. **Poor database session management**
4. **Ignoring security best practices**
5. **Not understanding the difference between path and query parameters**

### Must-Know Concepts

- Dependency injection system
- Pydantic model validation
- Request/response lifecycle
- Middleware implementation
- Background tasks vs Celery
- ASGI vs WSGI

---

_Good luck with your FastAPI interviews! Remember to emphasize practical experience and be ready to code live examples._
