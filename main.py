import os
import uuid
import sqlite3
import traceback
import time
import base64
import hashlib
import hmac
from typing import Literal, Optional, List

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from starlette.middleware.sessions import SessionMiddleware

from openai import OpenAI

# RAG
from kb.retrieve import retrieve

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "users.db")
EMAIL_LIST_PATH = os.path.join(DATA_DIR, "emails.txt")

# Admin token for downloading emails (set in Railway Variables)
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()

app = FastAPI(title="Abaqus AI Scripter")

SECRET_KEY = os.getenv("APP_SECRET_KEY", "dev-secret-change-me")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, same_site="lax", https_only=False)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


def append_email_list(email: str):
    try:
        existing = set()
        if os.path.exists(EMAIL_LIST_PATH):
            with open(EMAIL_LIST_PATH, "r", encoding="utf-8") as f:
                existing = set(x.strip().lower() for x in f if x.strip())

        e = email.strip().lower()
        if e and e not in existing:
            with open(EMAIL_LIST_PATH, "a", encoding="utf-8") as f:
                f.write(e + "\n")
    except Exception:
        pass


# -------------------- Password hashing (PBKDF2) --------------------
PBKDF2_ITERS = int(os.getenv("PBKDF2_ITERS", "260000"))


def hash_password(password: str) -> str:
    password_bytes = password.encode("utf-8")
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password_bytes, salt, PBKDF2_ITERS, dklen=32)
    salt_b64 = base64.b64encode(salt).decode("ascii")
    dk_b64 = base64.b64encode(dk).decode("ascii")
    return f"pbkdf2_sha256${PBKDF2_ITERS}${salt_b64}${dk_b64}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters_s, salt_b64, dk_b64 = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iters = int(iters_s)
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(dk_b64.encode("ascii"))
        test = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters, dklen=len(expected))
        return hmac.compare_digest(test, expected)
    except Exception:
        return False


# -------------------- Models --------------------
class GenerateRequest(BaseModel):
    mode: Literal["scripter", "debugger"] = "scripter"
    python_version: Literal["py2", "py3"] = "py3"
    prompt: str
    code: Optional[str] = None
    use_rag: bool = False
    rag_k: int = 6


class RetrieveRequest(BaseModel):
    query: str
    k: int = 6


class SignupRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


def is_logged_in(request: Request) -> bool:
    return bool(request.session.get("user_email"))


def require_login(request: Request) -> Optional[RedirectResponse]:
    if not is_logged_in(request):
        return RedirectResponse(url="/login", status_code=302)
    return None


# -------------------- Pages --------------------
@app.get("/")
def index(request: Request):
    guard = require_login(request)
    if guard:
        return guard
    return FileResponse("static/index.html")


@app.get("/login")
def login_page():
    return FileResponse("static/login.html")


@app.get("/signup")
def signup_page():
    return FileResponse("static/signup.html")


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)


# -------------------- Admin: download emails --------------------
@app.get("/admin/emails")
def admin_emails(request: Request):
    """
    Download the emails.txt file.
    Protect with a token:
      /admin/emails?token=YOUR_ADMIN_TOKEN
    """
    token = (request.query_params.get("token") or "").strip()

    if not ADMIN_TOKEN:
        return JSONResponse(
            {"ok": False, "error": "ADMIN_TOKEN not set on server."},
            status_code=500
        )

    if token != ADMIN_TOKEN:
        return JSONResponse({"ok": False, "error": "Unauthorized."}, status_code=401)

    if not os.path.exists(EMAIL_LIST_PATH):
        # Return an empty file to keep UX smooth
        os.makedirs(os.path.dirname(EMAIL_LIST_PATH), exist_ok=True)
        with open(EMAIL_LIST_PATH, "w", encoding="utf-8") as f:
            f.write("")

    return FileResponse(
        EMAIL_LIST_PATH,
        media_type="text/plain",
        filename="emails.txt",
    )


# -------------------- Auth API --------------------
@app.post("/api/signup")
def signup(req: SignupRequest, request: Request):
    try:
        password = (req.password or "").strip()
        if len(password) < 8:
            return JSONResponse({"ok": False, "error": "Password must be at least 8 characters."}, status_code=400)

        email = req.email.lower().strip()
        pw_hash = hash_password(password)

        conn = db_connect()
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
                (email, pw_hash, int(time.time())),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            return JSONResponse({"ok": False, "error": "Email already exists. Try logging in."}, status_code=400)
        finally:
            conn.close()

        append_email_list(email)

        request.session["user_email"] = email
        return JSONResponse({"ok": True})

    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": "Signup failed", "details": str(e) + "\n\n" + traceback.format_exc()},
            status_code=500,
        )


@app.post("/api/login")
def login(req: LoginRequest, request: Request):
    try:
        email = req.email.lower().strip()

        conn = db_connect()
        cur = conn.cursor()
        cur.execute("SELECT email, password_hash FROM users WHERE email = ?", (email,))
        row = cur.fetchone()
        conn.close()

        if not row:
            return JSONResponse({"ok": False, "error": "No account found for that email."}, status_code=400)

        if not verify_password((req.password or "").strip(), row["password_hash"]):
            return JSONResponse({"ok": False, "error": "Incorrect password."}, status_code=400)

        request.session["user_email"] = row["email"]
        return JSONResponse({"ok": True})

    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": "Login failed", "details": str(e) + "\n\n" + traceback.format_exc()},
            status_code=500,
        )


@app.get("/api/me")
def me(request: Request):
    return JSONResponse({"ok": True, "email": request.session.get("user_email")})


# -------------------- OpenAI helpers --------------------
def _client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError('OPENAI_API_KEY is not set. Use environment variable on Railway.')
    return OpenAI(api_key=key)


def _model_name() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _system_prompt(mode: str, python_version: str, rag_enabled: bool) -> str:
    py = "Python 2.7" if python_version == "py2" else "Python 3"
    base = (
        "You are an expert Abaqus/CAE Python scripting assistant.\n"
        "You write correct, runnable Abaqus scripts and explain decisions briefly.\n"
        "Always respect the requested Python version.\n"
        f"Target scripting version: {py}.\n"
    )
    if rag_enabled:
        base += (
            "You will be given REFERENCE SNIPPETS from Abaqus documentation.\n"
            "Rules:\n"
            "- Prefer methods/classes shown in REFERENCE SNIPPETS.\n"
            "- Do NOT invent API methods.\n"
            "- If you must choose between options, pick the one supported by the snippets.\n"
        )
    if mode == "scripter":
        return (
            base
            + "Output format:\n"
            "1) Assumptions (bullets)\n"
            "2) Plan (bullets)\n"
            "3) Complete Abaqus script (single code block)\n"
            "4) How to run (CAE vs noGUI)\n"
        )
    return (
        base
        + "You are in DEBUGGER mode.\n"
        "Output format:\n"
        "1) Likely root cause(s)\n"
        "2) Exact fixes (show corrected code snippets)\n"
        "3) Checklist to verify in Abaqus/CAE\n"
    )


def _format_snippets(snips: List[dict]) -> str:
    out = []
    for i, s in enumerate(snips, 1):
        text = (s.get("text") or "").strip()
        if len(text) > 900:
            text = text[:900] + " ...[truncated]"
        out.append(f"[{i}] source={s.get('source')} score={s.get('score'):.3f}\n{text}")
    return "\n\n".join(out)


# -------------------- RAG API (protected) --------------------
@app.post("/api/retrieve")
def api_retrieve(req: RetrieveRequest, request: Request):
    if not is_logged_in(request):
        return JSONResponse({"ok": False, "error": "Not authenticated. Please log in."}, status_code=401)
    try:
        hits = retrieve(req.query, k=max(1, min(20, int(req.k))))
        return JSONResponse({"ok": True, "hits": hits})
    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": "Retrieve failed", "details": str(e) + "\n\n" + traceback.format_exc()},
            status_code=500,
        )


# -------------------- Main AI API (protected, optional RAG) --------------------
from quality import validate_script, enforce_json, QualityError

@app.post("/api/generate")
def generate(req: GenerateRequest, request: Request):
    if not is_logged_in(request):
        return JSONResponse({"ok": False, "error": "Not authenticated."}, status_code=401)

    client = _client()
    model = _model_name()

    rag_enabled = bool(req.use_rag)
    snippets = []
    if rag_enabled:
        try:
            snippets = retrieve(req.prompt + " " + (req.code or ""), k=max(1, min(20, int(req.rag_k))))
        except Exception:
            snippets = []

    reference_block = _format_snippets(snippets) if snippets else ""

    # Hard rules + examples to prevent py2 violations
    rules = (
        "STRICT RULES (HARD FAIL IF VIOLATED):\n"
        "- If Python 2.7 selected: NO f-strings, NO print(), NO numpy, NO pathlib.\n"
        "- Use Python 2.7 print statements and %% formatting, e.g.:\n"
        "  print('x') is NOT allowed. Use: print 'x' or print \"x\"\n"
        "  Use formatting: print 'Max mises: %g' % max_mises\n"
        "- For ODB: use from odbAccess import openOdb; use frame.fieldOutputs['S']; use value.mises.\n"
        "- If element set is mentioned: MUST subset with getSubset(region=...).\n"
        "- Must search assembly elementSets AND ALL instances elementSets; do NOT use instances.keys()[0].\n"
        "- Do NOT invent Abaqus API methods.\n"
    )

    system_prompt = (
        _system_prompt(req.mode, req.python_version, rag_enabled)
        + "\n\n"
        + rules
        + "\n\n"
        + "Return ONLY valid JSON with keys: assumptions, plan, script, how_to_run.\n"
          "No markdown. No extra keys. No commentary outside JSON."
    )

    user_msg = (req.prompt or "").strip()
    if req.code:
        user_msg += "\n\nCODE:\n" + (req.code or "")

    if reference_block:
        user_msg += "\n\nREFERENCE SNIPPETS:\n" + reference_block

    def call_model(feedback=None, temperature=0.2):
        content = user_msg
        if feedback:
            content += "\n\nFIX THESE ISSUES EXACTLY:\n" + feedback
            content += "\n\nREMINDER (Python 2.7): no f-strings; no print(); use print 'text' and % formatting.\n"
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    # attempt 1
    raw = call_model(temperature=0.2)

    def try_validate(raw_text):
        data = enforce_json(raw_text)
        errors, warnings = validate_script(data["script"], req.python_version, req.prompt)
        return data, errors, warnings

    try:
        data, errors, warnings = try_validate(raw)
        if errors:
            raise QualityError("; ".join(errors))
        return JSONResponse({"ok": True, "result": data, "warnings": warnings, "rag_used": rag_enabled})
    except QualityError as e:
        # attempt 2 (repair, deterministic)
        raw2 = call_model(feedback=str(e), temperature=0.0)
        try:
            data2, errors2, warnings2 = try_validate(raw2)
            if errors2:
                raise QualityError("; ".join(errors2))
            return JSONResponse({"ok": True, "result": data2, "warnings": warnings2, "rag_used": rag_enabled, "repaired": True})
        except QualityError as e2:
            # attempt 3 (final repair, slightly more guidance)
            fb = str(e2) + "\n\nIf you used f-strings, replace them with % formatting. If you used print(), replace with print statements."
            raw3 = call_model(feedback=fb, temperature=0.0)
            try:
                data3, errors3, warnings3 = try_validate(raw3)
                if errors3:
                    return JSONResponse(
                        {"ok": False, "error": "Generation failed quality checks", "details": errors3, "warnings": warnings3, "raw_output": data3},
                        status_code=422,
                    )
                return JSONResponse({"ok": True, "result": data3, "warnings": warnings3, "rag_used": rag_enabled, "repaired": True})
            except Exception as e3:
                return JSONResponse({"ok": False, "error": "Generation failed after repair attempts", "details": str(e3)}, status_code=500)



@app.post("/api/upload")
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(...),
    chat_id: str = Form(default=""),
):
    if not is_logged_in(request):
        return JSONResponse({"ok": False, "error": "Not authenticated. Please log in."}, status_code=401)

    saved = []
    try:
        for f in files:
            orig_name = f.filename or "file"
            ext = os.path.splitext(orig_name)[1]
            safe_id = uuid.uuid4().hex
            stored_name = f"{safe_id}{ext}"
            stored_path = os.path.join(UPLOAD_DIR, stored_name)

            content = await f.read()
            with open(stored_path, "wb") as out:
                out.write(content)

            saved.append(
                {
                    "original_name": orig_name,
                    "stored_name": stored_name,
                    "size_bytes": len(content),
                    "url": f"/uploads/{stored_name}",
                    "chat_id": chat_id,
                }
            )

        return JSONResponse({"ok": True, "files": saved})

    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": "Upload failed", "details": str(e) + "\n\n" + traceback.format_exc()},
            status_code=500,
        )
