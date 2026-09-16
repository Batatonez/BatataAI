from __future__ import annotations

import base64
import html as html_lib
import json
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import unicodedata
import uuid
import webbrowser
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, unquote, urlparse

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
SETTINGS_PATH = ROOT / "data" / "settings.json"
UI_SETTINGS_PATH = ROOT / "data" / "ui-settings.json"
MEMORY_PATH = ROOT / "data" / "memory.json"
FOLDERS_PATH = ROOT / "data" / "folders.json"
PERSONA_PATH = ROOT / "persona" / "BatataAI-persona.txt"
CHATS_DIR = ROOT / "data" / "chats"
TRASH_DIR = ROOT / "data" / "trash"
ATTACHMENTS_DIR = ROOT / "data" / "attachments"
LOGS_DIR = ROOT / "logs"
WEB_DIR = ROOT / "web"
MODELS_DIR = ROOT / "models"
RUNTIME_DIR = ROOT / "runtime"

for p in (CHATS_DIR, TRASH_DIR, ATTACHMENTS_DIR, LOGS_DIR, MODELS_DIR, RUNTIME_DIR):
    p.mkdir(parents=True, exist_ok=True)

def load_json(path: Path, fallback):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[BatataAI] Aviso ao ler {path.name}: {e}")
    return fallback

def atomic_write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def load_persona():
    try:
        if PERSONA_PATH.exists():
            text = PERSONA_PATH.read_text(encoding="utf-8").strip()
            if text:
                return text
    except Exception as e:
        print(f"[BatataAI] Aviso ao ler persona: {e}")
    return ""

CFG = load_json(CONFIG_PATH, {
    "llamafile_port": 8080,
    "ui_port": 3210,
    "ctx_size": 0,
    "max_tokens": -1,
    "temperature": 0.7,
    "system_prompt": "",
    "memory_max_items": 40,
})

SETTINGS = load_json(SETTINGS_PATH, {"last_model": ""})

LLAMAFILE_PORT = int(CFG.get("llamafile_port", 8080))
UI_PORT = int(CFG.get("ui_port", 3210))
LLAMAFILE_URL = f"http://127.0.0.1:{LLAMAFILE_PORT}"

state_lock = threading.RLock()
llamafile_proc = None
llamafile_log = None
active_model = None
active_vision = False
active_mmproj = None
loading_model = False
last_error = ""


UI_SETTINGS_DEFAULT = {
    "theme": "graphite",
    "accent": "#40639c",
    "gradient_color": "#151923",
    "categories_collapsed": False,
    "gradients_enabled": True,
}

def _sanitize_ui_settings(raw):
    if not isinstance(raw, dict):
        raw = {}

    theme = str(raw.get("theme", UI_SETTINGS_DEFAULT["theme"])).strip().lower()
    if theme not in {"graphite", "midnight", "forest", "ember"}:
        theme = UI_SETTINGS_DEFAULT["theme"]

    accent = str(raw.get("accent", UI_SETTINGS_DEFAULT["accent"])).strip()
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", accent):
        accent = UI_SETTINGS_DEFAULT["accent"]

    gradient_color = str(
        raw.get("gradient_color", UI_SETTINGS_DEFAULT["gradient_color"])
    ).strip()
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", gradient_color):
        gradient_color = UI_SETTINGS_DEFAULT["gradient_color"]

    return {
        "theme": theme,
        "accent": accent.lower(),
        "gradient_color": gradient_color.lower(),
        "categories_collapsed": bool(
            raw.get("categories_collapsed", UI_SETTINGS_DEFAULT["categories_collapsed"])
        ),
        "gradients_enabled": bool(
            raw.get("gradients_enabled", UI_SETTINGS_DEFAULT["gradients_enabled"])
        ),
    }


def get_ui_settings():
    """
    UI settings live in their own file.

    Older BatataAI versions kept them inside data/settings.json together with
    last_model. That made a later model-setting write capable of replacing the
    UI state. We migrate that old value once and then keep the two concerns
    completely separate.
    """
    with state_lock:
        dedicated = load_json(UI_SETTINGS_PATH, None)
        if isinstance(dedicated, dict):
            return _sanitize_ui_settings(dedicated)

        # One-time migration from older versions.
        legacy_settings = load_json(SETTINGS_PATH, SETTINGS)
        legacy_ui = (
            legacy_settings.get("ui")
            if isinstance(legacy_settings, dict)
            else None
        )

        current = _sanitize_ui_settings(legacy_ui)

        # Persist immediately so the next server restart never depends on the
        # old shared settings file.
        atomic_write_json(UI_SETTINGS_PATH, current)
        return current


def save_ui_settings(data):
    with state_lock:
        current = get_ui_settings()

        if isinstance(data, dict):
            merged = dict(current)

            if "theme" in data:
                merged["theme"] = data.get("theme")

            if "accent" in data:
                merged["accent"] = data.get("accent")

            if "gradient_color" in data:
                merged["gradient_color"] = data.get("gradient_color")

            if "categories_collapsed" in data:
                merged["categories_collapsed"] = bool(data.get("categories_collapsed"))

            if "gradients_enabled" in data:
                merged["gradients_enabled"] = bool(data.get("gradients_enabled"))

            current = _sanitize_ui_settings(merged)

        atomic_write_json(UI_SETTINGS_PATH, current)
        return current


# -------------------------
# Persistent memory
# -------------------------

SENSITIVE_MEMORY_TERMS = (
    "senha", "password", "cpf", "rg ", "cartão", "cartao", "cvv",
    "token de acesso", "api key", "chave api", "secret key",
    "endereço completo", "endereco completo", "coordenadas",
    "diagnóstico", "diagnostico", "religião", "religiao",
    "partido político", "partido politico", "orientação sexual", "orientacao sexual",
)

MEMORY_TRIGGER_TERMS = (
    "sempre ", "sempre que", "daqui pra frente", "de agora em diante",
    "lembre que", "lembra que", "guarde que", "guarda que",
    "prefiro ", "eu prefiro", "não quero que você", "nao quero que voce",
    "quero que você sempre", "quero que voce sempre",
)

def load_memory():
    fallback = {
        "profile": {"name": "Davi"},
        "preferences": [
            "O nome do usuário é Davi. Não repita o nome dele em toda resposta; use apenas quando for natural."
        ],
        "facts": [],
        "updated_at": None,
    }
    mem = load_json(MEMORY_PATH, fallback)
    mem.setdefault("profile", {})
    mem.setdefault("preferences", [])
    mem.setdefault("facts", [])
    return mem

def save_memory(mem):
    mem["updated_at"] = datetime.now().isoformat(timespec="seconds")
    max_items = int(CFG.get("memory_max_items", 40))
    mem["preferences"] = mem.get("preferences", [])[-max_items:]
    mem["facts"] = mem.get("facts", [])[-max_items:]
    atomic_write_json(MEMORY_PATH, mem)

def memory_prompt():
    mem = load_memory()
    lines = ["MEMÓRIA PERSISTENTE DO USUÁRIO:"]
    name = str(mem.get("profile", {}).get("name", "")).strip()
    if name:
        lines.append(f"- Nome: {name}. Use apenas quando for natural; não repita em toda resposta.")
    for item in mem.get("preferences", []):
        item = str(item).strip()
        if item:
            lines.append(f"- Preferência: {item}")
    for item in mem.get("facts", []):
        item = str(item).strip()
        if item:
            lines.append(f"- Informação: {item}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)

def _contains_sensitive_memory(text: str) -> bool:
    low = text.lower()
    return any(term in low for term in SENSITIVE_MEMORY_TERMS)

def _clean_memory_sentence(text: str) -> str:
    clean = " ".join(text.replace("\n", " ").split()).strip()
    return clean[:360]

def maybe_update_memory(user_text: str):
    """
    Save only explicit, durable-looking user preferences/facts.
    This intentionally avoids storing every random chat message.
    """
    text = _clean_memory_sentence(user_text)
    if not text or _contains_sensitive_memory(text):
        return False

    low = text.lower()
    mem = load_memory()
    changed = False

    # Explicit name statement.
    m = re.search(r"\bmeu nome (?:é|e)\s+([A-Za-zÀ-ÖØ-öø-ÿ'-]{2,40})", text, re.I)
    if m:
        name = m.group(1).strip().title()
        if mem.get("profile", {}).get("name") != name:
            mem.setdefault("profile", {})["name"] = name
            changed = True

    # Explicit durable preference / reminder.
    if any(trigger in low for trigger in MEMORY_TRIGGER_TERMS):
        prefs = mem.setdefault("preferences", [])
        normalized = text.casefold()
        if all(str(p).casefold() != normalized for p in prefs):
            prefs.append(text)
            changed = True

    if changed:
        save_memory(mem)
    return changed

def memory_count():
    mem = load_memory()
    return len(mem.get("preferences", [])) + len(mem.get("facts", []))


# -------------------------
# Attachments
# -------------------------

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".py", ".js", ".jsx", ".ts", ".tsx",
    ".json", ".jsonl", ".csv", ".tsv", ".html", ".htm", ".css", ".scss",
    ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".log",
    ".bat", ".cmd", ".ps1", ".sh", ".zsh", ".fish", ".java", ".kt",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cs", ".go", ".rs", ".sql",
    ".env", ".gitignore", ".dockerfile", ".properties", ".gradle"
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_TEXT_CHARS_PER_FILE = 60_000
MAX_ATTACHMENT_TEXT_CHARS = 120_000

def _safe_filename(name: str) -> str:
    name = Path(str(name)).name
    name = re.sub(r"[^A-Za-z0-9À-ÖØ-öø-ÿ._()\- ]+", "_", name).strip(" .")
    return name[:160] or "arquivo"

def _attachment_path(attachment_id: str) -> Path | None:
    safe = re.sub(r"[^A-Za-z0-9\-]", "", str(attachment_id))
    matches = list(ATTACHMENTS_DIR.glob(f"{safe}__*"))
    return matches[0] if matches else None

def _extract_docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path, "r") as z:
            raw = z.read("word/document.xml")
        root = ET.fromstring(raw)
        texts = []
        for node in root.iter():
            if node.tag.endswith("}t") and node.text:
                texts.append(node.text)
            elif node.tag.endswith("}p"):
                texts.append("\n")
        return " ".join(texts).replace(" \n ", "\n").strip()
    except Exception:
        return ""

def _extract_pdf_text_optional(path: Path) -> str:
    # Remains dependency-free by default. If pypdf happens to exist on the PC,
    # BatataAI will use it automatically.
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        parts = []
        for page in reader.pages[:50]:
            parts.append(page.extract_text() or "")
        return "\n".join(parts).strip()
    except Exception:
        return ""

def extract_attachment_text(path: Path, original_name: str, mime: str) -> str:
    ext = Path(original_name).suffix.lower()

    if ext == ".docx":
        return _extract_docx_text(path)[:MAX_TEXT_CHARS_PER_FILE]

    if ext == ".pdf":
        return _extract_pdf_text_optional(path)[:MAX_TEXT_CHARS_PER_FILE]

    if ext in TEXT_EXTENSIONS or mime.startswith("text/"):
        raw = path.read_bytes()
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                return raw.decode(enc)[:MAX_TEXT_CHARS_PER_FILE]
            except Exception:
                continue

    return ""

def save_attachment(filename: str, mime: str, b64_data: str):
    safe_name = _safe_filename(filename)
    try:
        raw = base64.b64decode(b64_data, validate=True)
    except Exception:
        raise ValueError("Arquivo em Base64 inválido.")

    if not raw:
        raise ValueError("O arquivo está vazio.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("Arquivo grande demais. Limite atual: 25 MB por arquivo.")

    attachment_id = str(uuid.uuid4())
    target = ATTACHMENTS_DIR / f"{attachment_id}__{safe_name}"
    target.write_bytes(raw)

    guessed = mime or mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
    ext = target.suffix.lower()
    kind = "image" if guessed.startswith("image/") or ext in IMAGE_EXTENSIONS else "file"
    text = extract_attachment_text(target, safe_name, guessed)
    readable = bool(text)

    return {
        "id": attachment_id,
        "name": safe_name,
        "mime": guessed,
        "size": len(raw),
        "kind": kind,
        "readable_text": readable,
    }

def attachment_to_prompt_part(meta):
    path = _attachment_path(meta.get("id", ""))
    if not path or not path.exists():
        return None, f"[Anexo ausente: {meta.get('name', 'arquivo')}]"

    name = meta.get("name") or path.name.split("__", 1)[-1]
    mime = meta.get("mime") or mimetypes.guess_type(name)[0] or "application/octet-stream"
    kind = meta.get("kind") or ("image" if mime.startswith("image/") else "file")

    if kind == "image":
        # Visão está desativada por enquanto.
        # A imagem continua salva e visível na interface, mas seus pixels NÃO são
        # enviados ao modelo. Isso evita que um modelo tente adivinhar o conteúdo.
        return None, (
            f"[Imagem anexada: {name}. O conteúdo visual desta imagem NÃO está disponível "
            "para análise nesta configuração. Não tente adivinhar jogo, pessoa, objeto, texto, "
            "local, interface ou qualquer outro detalhe da imagem. Se a resposta depender do "
            "conteúdo visual, diga apenas que você não consegue confirmar pela imagem e peça "
            "uma descrição ou informação textual do usuário.]"
        )

    text = extract_attachment_text(path, name, mime)
    if text:
        return None, f'<arquivo_anexado nome="{name}">\n{text}\n</arquivo_anexado>'

    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        return None, (
            f"[PDF anexado: {name}. O conteúdo não pôde ser extraído localmente. "
            "Se pypdf estiver instalado, a BatataAI consegue extrair PDFs automaticamente.]"
        )
    return None, (
        f"[Arquivo anexado: {name} ({mime}). O formato foi armazenado, mas não possui "
        "extração de texto local configurada.]"
    )

def enrich_messages_with_attachments(chat_messages):
    result = []
    for msg in chat_messages:
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))
        attachments = msg.get("attachments") or []

        if role != "user" or not attachments:
            result.append({"role": role, "content": content})
            continue

        image_parts = []
        notes = []
        text_budget = MAX_ATTACHMENT_TEXT_CHARS

        for meta in attachments:
            part, note = attachment_to_prompt_part(meta)
            if part:
                image_parts.append(part)
            if note:
                if len(note) > text_budget:
                    note = note[:text_budget]
                text_budget -= len(note)
                notes.append(note)
                if text_budget <= 0:
                    break

        combined_text = content
        if notes:
            combined_text += ("\n\n" if combined_text else "") + "\n\n".join(notes)

        if image_parts:
            parts = [{"type": "text", "text": combined_text or "Analise os anexos enviados."}]
            parts.extend(image_parts)
            result.append({"role": role, "content": parts})
        else:
            result.append({"role": role, "content": combined_text})

    return result


# -------------------------
# Models / runtime
# -------------------------


def is_mmproj_file(path: Path) -> bool:
    low = path.name.lower()
    return "mmproj" in low or "projector" in low

def _vision_family_key(path: Path) -> str:
    """
    Produces a rough family key so we don't pair a model with an unrelated
    projector when multiple .gguf files live in the same folder.
    """
    stem = path.stem.lower()
    stem = re.sub(r"^(mmproj[-_.]*|projector[-_.]*)", "", stem)
    stem = re.sub(
        r"[-_.](q\d+(?:_[a-z0-9]+)*|iq\d+(?:_[a-z0-9]+)*|f16|f32|bf16|fp16|fp32)$",
        "",
        stem,
        flags=re.I,
    )
    return re.sub(r"[^a-z0-9]+", "", stem)

def first_installed_vision_model():
    models = [m for m in discover_models() if m.get("vision")]
    if not models:
        return None

    # Prefer models intentionally placed in models/vision/, then smaller ones.
    models.sort(
        key=lambda m: (
            0 if "/vision/" in ("/" + m["id"].replace("\\", "/").lower()) else 1,
            float(m.get("size_gb") or 9999),
            m.get("label", "").lower(),
        )
    )
    return models[0]

def attachments_have_images(attachments) -> bool:
    for meta in attachments or []:
        if not isinstance(meta, dict):
            continue
        kind = str(meta.get("kind", "")).lower()
        mime = str(meta.get("mime", "")).lower()
        if kind == "image" or mime.startswith("image/"):
            return True
    return False


def normalize_rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()

def discover_models():
    items = []
    for p in sorted(MODELS_DIR.rglob("*.gguf"), key=lambda x: str(x).lower()):
        if is_mmproj_file(p):
            continue

        rel = normalize_rel(p)
        lower = p.name.lower()

        if "ministral-3-3b-instruct-2512" in lower:
            label = "Ministral 3 3B — Visão"
        elif "q4_k_m" in lower and "qwen3" in lower and "4b" in lower:
            label = "Qwen3 4B — Q4_K_M"
        elif "q5_k_m" in lower and "qwen3" in lower and "4b" in lower:
            label = "Qwen3 4B — Q5_K_M"
        else:
            label = p.stem

        mmproj = find_mmproj_for_model(p)
        items.append({
            "id": rel,
            "filename": p.name,
            "label": label,
            "size_gb": round(p.stat().st_size / (1024**3), 2),
            "vision": bool(mmproj),
            "mmproj": mmproj.name if mmproj else None,
        })
    return items

def resolve_model(model_id: str) -> Path:
    candidate = (ROOT / model_id).resolve()
    models_root = MODELS_DIR.resolve()
    try:
        candidate.relative_to(models_root)
    except ValueError:
        raise ValueError("Modelo inválido.")
    if not candidate.exists() or candidate.suffix.lower() != ".gguf":
        raise FileNotFoundError("Arquivo GGUF não encontrado.")
    return candidate


def find_mmproj_for_model(model: Path):
    candidates = [
        p for p in model.parent.glob("*.gguf")
        if p.is_file() and is_mmproj_file(p)
    ]
    if not candidates:
        return None

    model_key = _vision_family_key(model)

    # Exact normalized family match first.
    exact = [p for p in candidates if _vision_family_key(p) == model_key]
    if exact:
        return sorted(exact, key=lambda p: p.name.lower())[0]

    # Many repositories use the full model stem inside the mmproj filename.
    stem = model.stem.lower()
    contains = [
        p for p in candidates
        if stem in p.stem.lower() or p.stem.lower().replace("mmproj-", "") in stem
    ]
    if contains:
        return sorted(contains, key=lambda p: p.name.lower())[0]

    # A folder dedicated to one vision model is safe to pair automatically.
    if len(candidates) == 1:
        non_projector_models = [
            p for p in model.parent.glob("*.gguf")
            if p.is_file() and not is_mmproj_file(p)
        ]
        if len(non_projector_models) == 1:
            return candidates[0]

    # Safer than silently loading a projector from another model.
    return None

def runtime_path():
    candidates = []
    if os.name == "nt":
        candidates = [RUNTIME_DIR / "llamafile.exe", RUNTIME_DIR / "llamafile"]
    else:
        candidates = [RUNTIME_DIR / "llamafile", RUNTIME_DIR / "llamafile.exe"]

    for p in candidates:
        if p.exists():
            if os.name != "nt":
                temp_copy = Path(tempfile.gettempdir()) / f"batataai-llamafile-{os.getuid()}"
                shutil.copy2(p, temp_copy)
                temp_copy.chmod(0o755)
                return temp_copy
            return p
    raise FileNotFoundError("llamafile não encontrado em runtime/llamafile.exe")

def wait_for_llamafile(proc, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{LLAMAFILE_URL}/v1/models", timeout=2) as r:
                if 200 <= r.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False

def stop_llamafile_locked():
    global llamafile_proc, llamafile_log, active_model, active_vision, active_mmproj
    proc = llamafile_proc
    llamafile_proc = None
    active_model = None
    active_vision = False
    active_mmproj = None

    if proc and proc.poll() is None:
        print("[BatataAI] Encerrando modelo atual...")
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    if llamafile_log:
        try:
            llamafile_log.close()
        except Exception:
            pass
        llamafile_log = None

def stop_llamafile():
    with state_lock:
        stop_llamafile_locked()

def start_model(model_id: str):
    global llamafile_proc, llamafile_log, active_model, active_vision, active_mmproj, loading_model, last_error

    with state_lock:
        if loading_model:
            raise RuntimeError("Já existe um modelo sendo carregado.")
        loading_model = True
        last_error = ""

    try:
        model = resolve_model(model_id)
        runtime = runtime_path()

        with state_lock:
            stop_llamafile_locked()

            log_path = LOGS_DIR / "llamafile.log"
            llamafile_log = open(log_path, "a", encoding="utf-8", buffering=1)

            mmproj = find_mmproj_for_model(model)
            args = [
                str(runtime),
                "-m", str(model),
                "--server",
                "--host", "127.0.0.1",
                "--port", str(LLAMAFILE_PORT),
                "--jinja",
                "--ctx-size", str(int(CFG.get("ctx_size", 0))),
            ]
            if mmproj:
                args.extend(["--mmproj", str(mmproj)])

            kwargs = {
                "cwd": str(ROOT),
                "stdout": llamafile_log,
                "stderr": subprocess.STDOUT,
            }
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

            print(f"[BatataAI] Carregando: {model.name}")
            proc = subprocess.Popen(args, **kwargs)
            llamafile_proc = proc

        if not wait_for_llamafile(proc):
            raise RuntimeError("llamafile não respondeu. Veja logs/llamafile.log")

        with state_lock:
            active_model = model_id
            active_mmproj = find_mmproj_for_model(model)
            active_vision = bool(active_mmproj)
            SETTINGS["last_model"] = model_id
            atomic_write_json(SETTINGS_PATH, SETTINGS)

        print(f"[BatataAI] Modelo pronto: {model.name}")
        return model.name

    except Exception as e:
        with state_lock:
            last_error = str(e)
        raise
    finally:
        with state_lock:
            loading_model = False

def server_status():
    with state_lock:
        alive = bool(llamafile_proc and llamafile_proc.poll() is None)
        return {
            "ok": True,
            "loaded": alive and bool(active_model),
            "loading": loading_model,
            "active_model": active_model,
            "active_vision": active_vision,
            "active_mmproj": active_mmproj.name if active_mmproj else None,
            "last_model": SETTINGS.get("last_model", ""),
            "last_error": last_error,
            "persona_loaded": bool(load_persona()),
            "memory_items": memory_count(),
            "ctx_size": int(CFG.get("ctx_size", 0)),
        }


# -------------------------
# Chat folders / archive
# -------------------------

def load_folders():
    data = load_json(FOLDERS_PATH, {"folders": []})
    folders = data.get("folders", []) if isinstance(data, dict) else []
    clean = []
    for item in folders:
        if not isinstance(item, dict):
            continue
        fid = str(item.get("id", "")).strip()
        name = str(item.get("name", "")).strip()
        if fid and name:
            clean.append({
                "id": fid,
                "name": name[:80],
                "created_at": item.get("created_at"),
            })
    return clean

def save_folders(folders):
    atomic_write_json(FOLDERS_PATH, {"folders": folders})

def create_folder(name: str):
    name = " ".join(str(name).split()).strip()
    if not name:
        raise ValueError("Digite um nome para a pasta.")
    if len(name) > 80:
        raise ValueError("O nome da pasta é grande demais.")

    folders = load_folders()
    if any(f["name"].casefold() == name.casefold() for f in folders):
        raise ValueError("Já existe uma pasta com esse nome.")

    folder = {
        "id": str(uuid.uuid4()),
        "name": name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    folders.append(folder)
    save_folders(folders)
    return folder

def rename_folder(folder_id: str, name: str):
    name = " ".join(str(name).split()).strip()
    if not name:
        raise ValueError("Digite um nome para a pasta.")
    if len(name) > 80:
        raise ValueError("O nome da pasta é grande demais.")

    folders = load_folders()
    target = None

    for folder in folders:
        if str(folder.get("id")) == str(folder_id):
            target = folder
            break

    if target is None:
        raise FileNotFoundError("Pasta não encontrada.")

    if any(
        str(f.get("id")) != str(folder_id)
        and str(f.get("name", "")).casefold() == name.casefold()
        for f in folders
    ):
        raise ValueError("Já existe uma pasta com esse nome.")

    target["name"] = name
    save_folders(folders)
    return target

def delete_folder(folder_id: str):
    folders = load_folders()
    before = len(folders)
    folders = [f for f in folders if f.get("id") != folder_id]
    if len(folders) == before:
        return False

    # Removing a category never deletes its chats; they become "Sem pasta".
    for p in CHATS_DIR.glob("*.json"):
        try:
            chat = json.loads(p.read_text(encoding="utf-8"))
            if chat.get("folder_id") == folder_id:
                chat["folder_id"] = None
                atomic_write_json(p, chat)
        except Exception:
            continue

    save_folders(folders)
    return True

def set_chat_folder(chat_id: str, folder_id):
    chat = load_chat(chat_id)
    if not chat:
        raise FileNotFoundError("Chat não encontrado.")

    if folder_id in ("", None):
        chat["folder_id"] = None
    else:
        folder_id = str(folder_id)
        if not any(f["id"] == folder_id for f in load_folders()):
            raise ValueError("Pasta não encontrada.")
        chat["folder_id"] = folder_id

    save_chat(chat)
    return chat

def rename_chat(chat_id: str, title: str):
    chat = load_chat(chat_id)
    if not chat:
        raise FileNotFoundError("Chat não encontrado.")

    title = " ".join(str(title).split()).strip()
    if not title:
        raise ValueError("Digite um título para o chat.")
    if len(title) > 120:
        raise ValueError("O título do chat é grande demais.")

    chat["title"] = title
    save_chat(chat)
    return chat

def set_chat_pinned(chat_id: str, pinned: bool):
    chat = load_chat(chat_id)
    if not chat:
        raise FileNotFoundError("Chat não encontrado.")

    chat["pinned"] = bool(pinned)
    save_chat(chat)
    return chat

def set_chat_archived(chat_id: str, archived: bool):
    chat = load_chat(chat_id)
    if not chat:
        raise FileNotFoundError("Chat não encontrado.")
    chat["archived"] = bool(archived)
    save_chat(chat)
    return chat

def _delete_chat_attachments(chat):
    # Attachments are kept while the chat is in Trash.
    # They are removed only on permanent deletion.
    for message in chat.get("messages", []):
        for meta in message.get("attachments", []) or []:
            path = _attachment_path(meta.get("id", ""))
            if path and path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass


def permanently_delete_chat(chat_id: str):
    chat = load_chat(chat_id)
    if not chat:
        raise FileNotFoundError("Chat não encontrado.")

    _delete_chat_attachments(chat)

    p = chat_path(chat_id)
    if p.exists():
        p.unlink()

    return True

def permanently_delete_trash(filename: str):
    safe = Path(filename).name
    src = TRASH_DIR / safe
    if not src.exists():
        raise FileNotFoundError("Chat apagado não encontrado.")

    try:
        chat = json.loads(src.read_text(encoding="utf-8"))
        _delete_chat_attachments(chat)
    except Exception:
        pass

    src.unlink()
    return True


# -------------------------
# Chat persistence / trash
# -------------------------

def chat_path(chat_id: str) -> Path:
    safe = "".join(c for c in chat_id if c.isalnum() or c in "-_")
    return CHATS_DIR / f"{safe}.json"

def new_chat():
    now = datetime.now().isoformat(timespec="seconds")
    chat = {
        "id": str(uuid.uuid4()),
        "title": "Novo chat",
        "created_at": now,
        "updated_at": now,
        "messages": [],
        "folder_id": None,
        "archived": False,
        "pinned": False,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    atomic_write_json(chat_path(chat["id"]), chat)
    return chat

def load_chat(chat_id: str):
    p = chat_path(chat_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))

def save_chat(chat):
    chat["updated_at"] = datetime.now().isoformat(timespec="seconds")
    atomic_write_json(chat_path(chat["id"]), chat)

def list_chats():
    result = []

    for p in CHATS_DIR.glob("*.json"):
        try:
            chat = json.loads(p.read_text(encoding="utf-8"))

            search_parts = [str(chat.get("title", ""))]
            for message in chat.get("messages", []) or []:
                if not isinstance(message, dict):
                    continue
                content = str(message.get("content", "") or "").strip()
                if content:
                    search_parts.append(content)

            # Enough for local search without sending giant JSON responses for
            # extremely long conversations.
            search_text = "\n".join(search_parts)[-24000:]

            result.append({
                "id": chat.get("id"),
                "title": chat.get("title", "Chat"),
                "updated_at": chat.get("updated_at"),
                "folder_id": chat.get("folder_id"),
                "archived": bool(chat.get("archived", False)),
                "pinned": bool(chat.get("pinned", False)),
                "search_text": search_text,
            })
        except Exception:
            continue

    # Most recently updated first, but pinned chats always stay above the
    # others inside their section/folder.
    result.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
    result.sort(key=lambda x: bool(x.get("pinned", False)), reverse=True)
    return result

def soft_delete_chat(chat_id: str):
    src = chat_path(chat_id)
    if not src.exists():
        return False
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = TRASH_DIR / f"{src.stem}__{stamp}.json"
    shutil.move(str(src), str(dst))
    return True

def list_trash():
    result = []
    for p in sorted(TRASH_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            chat = json.loads(p.read_text(encoding="utf-8"))
            result.append({
                "trash_file": p.name,
                "id": chat.get("id"),
                "title": chat.get("title", "Chat apagado"),
                "deleted_at": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
            })
        except Exception:
            continue
    return result

def restore_trash(filename: str):
    safe = Path(filename).name
    src = TRASH_DIR / safe
    if not src.exists():
        raise FileNotFoundError("Item da lixeira não encontrado.")
    chat = json.loads(src.read_text(encoding="utf-8"))
    cid = chat.get("id") or str(uuid.uuid4())
    chat["id"] = cid
    chat["archived"] = False
    save_chat(chat)
    src.unlink()
    return chat

# -------------------------
# llama.cpp / llamafile API helpers
# -------------------------


# -------------------------
# Time / web context
# -------------------------

SAO_PAULO_TZ = timezone(timedelta(hours=-3), name="BRT")

WEB_TRIGGER_RE = re.compile(
    r"\b("
    r"pesquis(?:a|e|ar|e na web|a na web)|procura(?:r| na web)?|busca(?:r| na web)?|"
    r"web|internet|online|site|fonte|fontes|link|links|"
    r"hoje|agora|atual|atualmente|recente|recentes|últim[oa]s?|ultim[oa]s?|"
    r"not[ií]cia(?:s)?|lançamento|lançamento|vers[aã]o mais nova|"
    r"preço|precos|preços|cotação|cota[cç][aã]o|"
    r"clima|previs[aã]o do tempo|temperatura|"
    r"quem [ée] |o que aconteceu|quando vai|quando sai"
    r")\b",
    re.I,
)

# Searches are sent with strict Safe Search. This additional guard prevents
# obviously unsuitable or dangerous lookups from being forwarded by the
# automatic web-search layer.
WEB_BLOCK_RE = re.compile(
    r"\b("
    r"comprar arma|onde comprar arma|muni[cç][aã]o|explosivo|bomba caseira|"
    r"comprar droga|onde comprar droga|vape|cigarro|nicotina|"
    r"casa de aposta|site de aposta|apostas online|cassino online"
    r")\b",
    re.I,
)

def sao_paulo_now():
    return datetime.now(SAO_PAULO_TZ)

def runtime_context_prompt():
    now = sao_paulo_now()
    weekday = [
        "segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
        "sexta-feira", "sábado", "domingo"
    ][now.weekday()]
    return (
        "CONTEXTO ATUAL DO SISTEMA\n"
        f"- Local padrão do usuário: São Paulo, SP, Brasil.\n"
        f"- Data atual em São Paulo: {now.strftime('%d/%m/%Y')} ({weekday}).\n"
        f"- Hora atual em São Paulo: {now.strftime('%H:%M:%S')} BRT (UTC-3).\n"
        "- Você TEM acesso a essa data e hora por meio deste contexto. "
        "Não diga que não possui relógio quando a pergunta puder ser respondida com estes dados."
    )

def _strip_html(value):
    value = re.sub(r"<[^>]+>", " ", str(value or ""))
    value = html_lib.unescape(value)
    return re.sub(r"\s+", " ", value).strip()

def _unwrap_ddg_url(url):
    url = html_lib.unescape(str(url or "")).strip()
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        if "uddg" in params and params["uddg"]:
            return unquote(params["uddg"][0])
    except Exception:
        pass
    return url

EXTERNAL_SCHEDULE_RE = re.compile(
    r"\b(?:"
    r"que dia(?: que)? (?:vai ser|sera|acontece|acontecera|comeca|comecara)|"
    r"qual(?: e)? (?:o )?dia (?:da|do|de)|"
    r"qual(?: e)? (?:a )?data (?:da|do|de)|"
    r"quando (?:vai ser|sera|acontece|acontecera|comeca|comecara|abre|abrira)|"
    r"que horas (?:a|o|as|os|essa|esse|este|esta|isso|evento|show|feira|jogo|partida)|"
    r"que horas .*? (?:comeca|comecam|abre|abrem|acontece|acontecem)|"
    r"qual(?: e)? (?:o )?horario (?:da|do|de)"
    r")\b",
    re.I,
)

LOCAL_NOW_RE = re.compile(
    r"^\s*(?:batata(?:ai)?[\s,:-]*)?(?:"
    r"que horas(?: sao)?|qual(?: e)? a hora|que hora e|hora agora|horario agora|"
    r"que dia e hoje|qual(?: e)? a data(?: de hoje)?|data de hoje|qual dia e hoje|dia de hoje"
    r")\s*[?!.,]*$",
    re.I,
)

def is_external_schedule_query(text):
    normalized = normalize_user_text(text)
    if not normalized:
        return False

    # "que horas são?" / "qual a data de hoje?" são utilitários locais.
    if LOCAL_NOW_RE.fullmatch(normalized):
        return False

    return bool(EXTERNAL_SCHEDULE_RE.search(normalized))

def should_search_web(text):
    text = str(text or "").strip()
    if not text or WEB_BLOCK_RE.search(text):
        return False

    normalized = normalize_user_text(text)

    # Data/hora local vem do relógio do próprio BatataAI.
    if LOCAL_NOW_RE.fullmatch(normalized):
        return False

    # Data/horário de evento, lançamento, feira, partida etc. é informação
    # externa e deve ser pesquisada.
    if is_external_schedule_query(text):
        return True

    return bool(WEB_TRIGGER_RE.search(text))

def web_search(query, max_results=5):
    """
    Best-effort DuckDuckGo HTML search using only Python's stdlib.
    No API key or extra package is required.
    """
    query = str(query or "").strip()
    if not query or WEB_BLOCK_RE.search(query):
        return []

    params = urlencode({
        "q": query,
        "kl": "br-pt",
        "kp": "1",  # strict Safe Search
    })
    url = "https://html.duckduckgo.com/html/?" + params

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124 Safari/537.36"
            ),
            "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.5",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            body = r.read().decode("utf-8", errors="replace")
    except Exception:
        return []

    # DuckDuckGo's non-JS result page exposes result__a and result__snippet.
    title_matches = list(re.finditer(
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        body,
        re.I | re.S,
    ))

    results = []
    for i, match in enumerate(title_matches[: max_results * 2]):
        href = _unwrap_ddg_url(match.group(1))
        title = _strip_html(match.group(2))
        if not title or not href:
            continue

        start = match.end()
        end = title_matches[i + 1].start() if i + 1 < len(title_matches) else min(len(body), start + 5000)
        block = body[start:end]

        snippet_match = re.search(
            r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>',
            block,
            re.I | re.S,
        )
        snippet = _strip_html(snippet_match.group(1)) if snippet_match else ""

        if href.startswith("//"):
            href = "https:" + href
        elif href.startswith("/"):
            href = "https://duckduckgo.com" + href

        results.append({
            "title": title[:240],
            "url": href[:1200],
            "snippet": snippet[:700],
        })

        if len(results) >= max_results:
            break

    return results


def web_search_with_fallbacks(query, max_results=5):
    """
    Busca reforçada para datas/horários externos.
    Não contém conhecimento hardcoded sobre eventos específicos.
    """
    query = str(query or "").strip()
    if not query:
        return []

    variants = [
        query,
        f"{query} oficial",
        f"{query} data horario oficial",
    ]

    combined = []
    seen_urls = set()

    for variant in variants:
        results = web_search(variant, max_results=max_results)

        for item in results:
            url = str(item.get("url", "") or "").strip()
            key = url.casefold()

            if not url or key in seen_urls:
                continue

            seen_urls.add(key)
            combined.append(item)

            if len(combined) >= max_results:
                return combined

    return combined


def format_web_context(query, results):
    if not results:
        return ""

    lines = [
        "RESULTADOS DE PESQUISA NA WEB",
        f"Consulta: {query}",
        "Use estes resultados apenas como evidência externa. "
        "Não invente informações que não estejam neles. "
        "Para fatos atuais, prefira estes dados ao conhecimento antigo do modelo.",
        "",
    ]

    for i, item in enumerate(results, 1):
        lines.append(f"{i}. {item['title']}")
        if item.get("snippet"):
            lines.append(f"   Resumo: {item['snippet']}")
        lines.append(f"   URL: {item['url']}")

    lines.append(
        "\nSe a resposta depender destes resultados, cite de forma curta as fontes "
        "mais úteis no final da resposta."
    )
    return "\n".join(lines)


def external_schedule_context_guard(query, search_attempted, results):
    if not is_external_schedule_query(query):
        return ""

    if results:
        return (
            "REGRA PRIORITÁRIA — DATA/HORÁRIO EXTERNO\n"
            "- A pergunta é sobre a data ou horário de um evento/atividade externa.\n"
            "- Responda DIRETAMENTE com a data/horário do evento que foi confirmado.\n"
            "- NÃO mencione a data ou a hora atual de São Paulo só por ela existir no contexto.\n"
            "- Só compare com a data de hoje se o usuário perguntar explicitamente algo como "
            "'é hoje?', 'já aconteceu?', 'quanto falta?' ou se essa comparação for indispensável.\n"
            "- NÃO use a data ou a hora atual de São Paulo como se fosse a data/horário do evento.\n"
            "- A pesquisa na web retornou resultados: use-os como base.\n"
            "- Não diga que não possui acesso a informações externas quando resultados "
            "de pesquisa estiverem presentes.\n"
            "- Prefira fonte oficial quando houver e não invente o que não estiver confirmado."
        )

    if search_attempted:
        return (
            "REGRA PRIORITÁRIA — DATA/HORÁRIO EXTERNO\n"
            "- A pergunta é sobre a data ou horário de um evento/atividade externa.\n"
            "- A pesquisa foi tentada, mas não retornou resultado utilizável.\n"
            "- NÃO mencione a data/hora atual como preenchimento ou contexto desnecessário.\n"
            "- NÃO responda com a data/hora atual de São Paulo como se fosse a do evento.\n"
            "- NÃO invente data ou horário. Diga apenas que não conseguiu confirmar a informação."
        )

    return ""




SIMPLE_GREETING_RE = re.compile(
    r"^\s*(?:"
    r"oi+|ol[aá]+|opa+|opaa+|salve+|salvee+|e+a[ií]+|eae+|fala+|"
    r"yo+|hey+|bom dia+|boa tarde+|boa noite+"
    r")(?:\s+batata+a*)?[!?.\s]*$",
    re.I,
)

def is_simple_greeting(text):
    text = str(text or "").strip()
    return bool(text and SIMPLE_GREETING_RE.fullmatch(text))

def simple_greeting_guard():
    return (
        "MODO DE SAUDAÇÃO SIMPLES\n"
        "- A mensagem atual é apenas uma saudação casual.\n"
        "- Responda com uma saudação curta, natural e coerente com o tom do usuário.\n"
        "- Não invente contexto compartilhado, piada interna, segredo, referência anterior "
        "ou algo que 'só vocês sabem' se o usuário não mencionou isso.\n"
        "- Não se apresente de novo e não explique que é a BatataAI.\n"
        "- Não acrescente oferta genérica de ajuda.\n"
        "- Exemplos de estilo aceitável: 'Salveee!', 'Opa!', 'Eae!', 'Falaaa!'.\n"
        "- Não copie os exemplos mecanicamente; apenas siga esse nível de simplicidade."
    )



def normalize_user_text(text):
    text = str(text or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text)
    return text

CURRENT_TIME_CLAUSE_RE = re.compile(
    r"^(?:"
    r"(?:batata(?:ai)?[\s,:-]*)?"
    r"(?:que horas(?: sao)?|qual(?: e)? a hora|que hora e|"
    r"me fala a hora|fala a hora|hora agora|horario agora)"
    r"(?:\s+agora)?"
    r"(?:\s+(?:aqui|por aqui|em sao paulo|em sp))?"
    r")\s*[?!.,]*$",
    re.I,
)

CURRENT_DATE_CLAUSE_RE = re.compile(
    r"^(?:"
    r"(?:batata(?:ai)?[\s,:-]*)?"
    r"(?:que dia e hoje|qual(?: e)? a data|data de hoje|"
    r"qual dia e hoje|dia de hoje)"
    r")\s*[?!.,]*$",
    re.I,
)

TIME_FOLLOWUP_RE = re.compile(
    r"^(?:e\s+)?(?:agora|agr|e agora|e agr)\s*[?!.,]*$",
    re.I,
)

def _last_question_clause(text):
    """
    Uses the last question-like clause instead of matching any occurrence of
    'que horas'. This lets:
      'eae, só na paz? que horas sao?' -> current local time
    while:
      'que horas a BGS 2026 começa?' -> NOT local current time
    """
    normalized = normalize_user_text(text)
    if not normalized:
        return ""

    # Split greeting/small-talk before the actual question.
    parts = [
        part.strip(" ,;:-")
        for part in re.split(r"[?!]+", normalized)
        if part.strip(" ,;:-")
    ]
    return parts[-1] if parts else normalized.strip(" ,;:-")

def _message_is_explicit_local_time(text):
    clause = _last_question_clause(text)
    return bool(clause and CURRENT_TIME_CLAUSE_RE.fullmatch(clause))

def _message_is_explicit_local_date(text):
    clause = _last_question_clause(text)
    return bool(clause and CURRENT_DATE_CLAUSE_RE.fullmatch(clause))

def _previous_context_was_local_time(chat_messages):
    """
    Follow-ups like 'e agora?' only inherit the time intent if the immediately
    preceding exchange was really about the current local time.
    """
    messages = list(chat_messages or [])

    # Strongest signal: our deterministic assistant answer.
    for msg in reversed(messages[-4:]):
        if msg.get("role") == "assistant":
            assistant_text = normalize_user_text(msg.get("content", ""))
            if re.search(
                r"\bagora sao \d{1,2}:\d{2}(?::\d{2})? em sao paulo\b",
                assistant_text,
            ):
                return "time"
            if re.search(r"\bhoje e .*\d{2}/\d{2}/\d{4}\b", assistant_text):
                return "date"
            break

    # Fallback: inspect the most recent user question.
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        previous = str(msg.get("content", "") or "")
        if _message_is_explicit_local_time(previous):
            return "time"
        if _message_is_explicit_local_date(previous):
            return "date"
        break

    return None

def detect_local_time_intent(current_text, chat_messages=None):
    text = normalize_user_text(current_text)

    if _message_is_explicit_local_time(text):
        return "time"

    if _message_is_explicit_local_date(text):
        return "date"

    # Contextual follow-up only. "e agora?" by itself is NOT automatically
    # a time request; it inherits the immediately preceding time/date topic.
    if TIME_FOLLOWUP_RE.fullmatch(text):
        return _previous_context_was_local_time(chat_messages)

    return None

def greeting_prefix(text):
    normalized = normalize_user_text(text)
    if normalized.startswith(("salve", "salvee")):
        return "Salve! "
    if normalized.startswith(("eae", "eai", "e ai")):
        return "Eae! "
    if normalized.startswith(("opa", "opaa")):
        return "Opa! "
    if normalized.startswith(("oi", "ola")):
        return "Oi! "
    if normalized.startswith(("fala", "yo")):
        return "Fala! "
    return ""

def local_time_answer(kind, current_text=""):
    now = sao_paulo_now()
    weekdays = [
        "segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
        "sexta-feira", "sábado", "domingo",
    ]
    prefix = greeting_prefix(current_text)

    if kind == "date":
        return (
            f"{prefix}Hoje é {weekdays[now.weekday()]}, "
            f"{now.strftime('%d/%m/%Y')}."
        )

    return f"{prefix}Agora são {now.strftime('%H:%M:%S')} em São Paulo."



# -------------------------
# PT-BR output cleanup
# -------------------------

PTBR_WORD_REPLACEMENTS = {
    "jobs": "empregos",
    "job": "emprego",
    "grading": "correção",
    "else": "mais",
    "anyway": "enfim",
    "actually": "na verdade",
    "something": "algo",
    "whatever": "tanto faz",
    "skills": "habilidades",
    "skill": "habilidade",
    "workflow": "fluxo de trabalho",
    "workflows": "fluxos de trabalho",
    "task": "tarefa",
    "tasks": "tarefas",
    "tips": "dicas",
    "tip": "dica",
    "overview": "visão geral",
    "summary": "resumo",
    "example": "exemplo",
    "examples": "exemplos",
    "pros": "vantagens",
    "cons": "desvantagens",
}

def _preserve_case_replacement(original, replacement):
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement

def _cleanup_ptbr_plain_segment(text):
    text = re.sub(r"\balgo\s+else\b", "algo mais", text, flags=re.I)

    for english, portuguese in PTBR_WORD_REPLACEMENTS.items():
        pattern = re.compile(rf"\b{re.escape(english)}\b", re.I)

        def repl(match):
            return _preserve_case_replacement(match.group(0), portuguese)

        text = pattern.sub(repl, text)

    return text

def normalize_ptbr_output(text):
    """
    Final language lint for visible prose.
    It skips fenced code, inline code and raw URLs.
    """
    text = str(text or "")
    if not text:
        return text

    fenced_parts = re.split(r"(```[\s\S]*?```)", text)

    for i, fenced_part in enumerate(fenced_parts):
        if fenced_part.startswith("```"):
            continue

        inline_parts = re.split(r"(`[^`\n]+`)", fenced_part)

        for j, inline_part in enumerate(inline_parts):
            if inline_part.startswith("`"):
                continue

            url_parts = re.split(r"(https?://[^\s)>\]]+)", inline_part, flags=re.I)

            for k, url_part in enumerate(url_parts):
                if re.fullmatch(r"https?://[^\s)>\]]+", url_part, flags=re.I):
                    continue
                url_parts[k] = _cleanup_ptbr_plain_segment(url_part)

            inline_parts[j] = "".join(url_parts)

        fenced_parts[i] = "".join(inline_parts)

    return "".join(fenced_parts)


def final_behavior_guard():
    now = sao_paulo_now()
    return (
        "REGRAS PRIORITÁRIAS PARA ESTA RESPOSTA\n"
        "1. Responda em português brasileiro. Todo texto visível, inclusive títulos, "
        "subtítulos, listas e rótulos, deve ficar em português brasileiro, exceto nomes próprios, "
        "código ou termos técnicos realmente necessários.\n"
        "2. Não use palavras inglesas quando houver equivalente natural em português. "
        "Exemplos proibidos em texto comum: 'Jobs', 'Grading', 'else', 'anyway', 'actually', "
        "'whatever', 'something', 'skills', 'tasks', 'tips', 'overview' e 'summary'. "
        "Traduza esses termos antes de responder.\n"
        "3. Se mensagens anteriores do assistente disserem que você não possui relógio, "
        "não possui hora atual ou não sabe a hora, ignore essa afirmação: ela está errada.\n"
        f"4. A hora atual confiável em São Paulo é {now.strftime('%H:%M:%S')} "
        f"e a data é {now.strftime('%d/%m/%Y')}.\n"
        "5. Se o usuário perguntar a hora/data atual, responda diretamente usando esses dados. "
        "Não mande o usuário olhar celular, computador ou relógio.\n"
        "6. Não termine respostas simples com ofertas genéricas como "
        "'se quiser, posso ajudar com algo mais' sem necessidade.\n"
        "7. Não invente intimidade, segredo, piada interna ou contexto compartilhado "
        "que o usuário não mencionou explicitamente.\n"
        "8. Se o usuário perguntar a data ou horário de um evento, lançamento, feira, jogo "
        "ou outra coisa externa, responda sobre ESSA coisa. Não acrescente a data/hora de hoje "
        "sem necessidade. Só compare com o momento atual se o usuário pedir ou se isso for "
        "essencial para responder."
    )


def build_system_messages(chat_messages, web_context=""):
    payload_messages = []
    persona = load_persona()
    fallback = str(CFG.get("system_prompt", "")).strip()
    system_prompt = persona if persona else fallback

    mem = memory_prompt()
    if mem:
        system_prompt = (system_prompt + "\n\n" + mem).strip()

    system_prompt = (
        system_prompt
        + "\n\n"
        + runtime_context_prompt()
    ).strip()

    if web_context:
        system_prompt = (
            system_prompt
            + "\n\n"
            + web_context
        ).strip()

    # The latest user message can receive a tiny turn-specific guard.
    latest_user_text = ""
    for msg in reversed(chat_messages):
        if msg.get("role") == "user":
            latest_user_text = str(msg.get("content", "") or "")
            break

    if is_simple_greeting(latest_user_text):
        system_prompt = (
            system_prompt
            + "\n\n"
            + simple_greeting_guard()
        ).strip()

    # Keep these rules last so small local models don't lose them among
    # persona, memory and web-search context.
    system_prompt = (
        system_prompt
        + "\n\n"
        + final_behavior_guard()
    ).strip()

    if system_prompt:
        payload_messages.append({"role": "system", "content": system_prompt})
    payload_messages.extend(enrich_messages_with_attachments(chat_messages))
    return payload_messages

def post_json(url: str, payload, timeout=30):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

def tokenize_text(text: str) -> int:
    if not text:
        return 0
    try:
        result = post_json(
            f"{LLAMAFILE_URL}/tokenize",
            {"content": text, "add_special": False, "parse_special": True},
            timeout=30,
        )
        return len(result.get("tokens", []))
    except Exception:
        # Fallback is only an estimate.
        return max(1, round(len(text) / 3.8))

def prompt_token_count(messages) -> int:
    try:
        templated = post_json(
            f"{LLAMAFILE_URL}/apply-template",
            {"messages": messages},
            timeout=30,
        )
        prompt = str(templated.get("prompt", ""))
        return tokenize_text(prompt)
    except Exception:
        # Safe fallback estimate.
        chunks = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        chunks.append(str(part.get("text", "")))
            else:
                chunks.append(str(content))
        text = "\n".join(chunks)
        return max(1, round(len(text) / 3.8))

def local_chat_completion(messages, *, max_tokens=None, temperature=None):
    st = server_status()
    if not st["loaded"]:
        raise RuntimeError("Nenhum modelo está carregado.")

    payload = {
        "model": "LLaMA_CPP",
        "messages": messages,
        "temperature": float(CFG.get("temperature", 0.7) if temperature is None else temperature),
        "max_tokens": int(CFG.get("max_tokens", -1) if max_tokens is None else max_tokens),
        "stream": False,
    }

    req = urllib.request.Request(
        f"{LLAMAFILE_URL}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer no-key"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            result = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llamafile HTTP {e.code}: {body}")

    try:
        return str(result["choices"][0]["message"]["content"]).strip()
    except Exception:
        raise RuntimeError("O modelo retornou uma resposta em formato inesperado.")

def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
    text = re.sub(r"<think>.*$", "", text, flags=re.I | re.S)
    return text.strip()

def _fallback_title(user_text: str) -> str:
    clean = " ".join(user_text.replace("\n", " ").split()).strip()
    low = clean.lower().strip("!?., ")
    greetings = {
        "oi", "ola", "olá", "opa", "opaa", "opaaa", "opaaaa",
        "eae", "e aí", "eai", "yo", "hello", "hi", "hey"
    }
    if low in greetings or len(low) <= 2:
        return "Conversa casual"
    words = clean.split()
    title = " ".join(words[:6])
    if len(title) > 48:
        title = title[:47].rstrip() + "…"
    return title or "Novo chat"

def generate_chat_title(messages) -> str:
    first_user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    first_assistant = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")

    prompt = [
        {
            "role": "system",
            "content": (
                "Você cria títulos curtos para conversas. Descubra o assunto principal em vez "
                "de copiar literalmente a primeira mensagem. Responda SOMENTE com um título "
                "natural de 2 a 6 palavras, no idioma da conversa. Sem aspas, emoji, ponto final "
                "ou prefixo 'Título:'. Se for apenas saudação, responda exatamente: Conversa casual."
            ),
        },
        {
            "role": "user",
            "content": (
                "/no_think\n"
                f"Primeira mensagem:\n{first_user}\n\n"
                f"Primeira resposta:\n{first_assistant}\n\n"
                "Crie o título."
            ),
        },
    ]

    try:
        raw = local_chat_completion(prompt, max_tokens=48, temperature=0.2)
        title = _strip_thinking(raw).splitlines()[-1].strip()
        title = re.sub(r"^(t[ií]tulo|title)\s*:\s*", "", title, flags=re.I)
        title = title.strip(" \t\r\n\"'`*_#.:;!?-")
        title = " ".join(title.split())
        if not title or len(title) > 60:
            return _fallback_title(first_user)
        if len(title.split()) > 8:
            title = " ".join(title.split()[:8])
        if len(title) > 48:
            title = title[:47].rstrip() + "…"
        return title
    except Exception as e:
        print(f"[BatataAI] Aviso ao gerar título: {e}")
        return _fallback_title(first_user)

# -------------------------
# HTTP API
# -------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "BatataAI/2.3"

    def log_message(self, fmt, *args):
        return

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, content_type: str):
        if not path.exists():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def sse(self, event, data):
        payload = json.dumps(data, ensure_ascii=False)
        self.wfile.write(f"event: {event}\ndata: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    def do_GET(self):
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            return self.send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")

        if path == "/api/status":
            return self.send_json(server_status())

        if path == "/api/ui-settings":
            return self.send_json(get_ui_settings())

        if path == "/api/models":
            return self.send_json(discover_models())

        if path == "/api/chats":
            return self.send_json(list_chats())

        if path == "/api/folders":
            return self.send_json(load_folders())

        if path == "/api/memory":
            return self.send_json(load_memory())

        if path == "/api/trash":
            return self.send_json(list_trash())

        if path.startswith("/api/attachments/"):
            attachment_id = path.split("/")[-1]
            file_path = _attachment_path(attachment_id)
            if not file_path or not file_path.exists():
                return self.send_error(404)
            content_type = mimetypes.guess_type(file_path.name.split("__", 1)[-1])[0] or "application/octet-stream"
            return self.send_file(file_path, content_type)

        if path.startswith("/api/chats/"):
            chat_id = path.split("/")[-1]
            chat = load_chat(chat_id)
            if chat is None:
                return self.send_json({"error": "Chat não encontrado"}, 404)
            return self.send_json(chat)

        self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path


        if path == "/api/ui-settings":
            try:
                body = self.read_json()
                return self.send_json(save_ui_settings(body))
            except Exception as e:
                return self.send_json({"error": str(e)}, 400)

        if path == "/api/attachments":
            try:
                body = self.read_json()
                filename = str(body.get("filename", "arquivo"))
                mime = str(body.get("mime", "application/octet-stream"))
                data = str(body.get("data", ""))
                meta = save_attachment(filename, mime, data)
                return self.send_json(meta, 201)
            except Exception as e:
                return self.send_json({"error": str(e)}, 400)


        if path == "/api/folders":
            try:
                body = self.read_json()
                folder = create_folder(str(body.get("name", "")))
                return self.send_json(folder, 201)
            except Exception as e:
                return self.send_json({"error": str(e)}, 400)

        if path.startswith("/api/folders/") and path.endswith("/rename"):
            try:
                parts = path.strip("/").split("/")
                if len(parts) != 4:
                    return self.send_json({"error": "Rota inválida"}, 404)
                body = self.read_json()
                folder = rename_folder(parts[2], body.get("name", ""))
                return self.send_json({"ok": True, "folder": folder})
            except FileNotFoundError as e:
                return self.send_json({"error": str(e)}, 404)
            except Exception as e:
                return self.send_json({"error": str(e)}, 400)

        if path.startswith("/api/chats/") and path.endswith("/archive"):
            try:
                parts = path.strip("/").split("/")
                chat = set_chat_archived(parts[2], bool(self.read_json().get("archived", True)))
                return self.send_json({"ok": True, "chat": chat})
            except Exception as e:
                return self.send_json({"error": str(e)}, 400)

        if path.startswith("/api/chats/") and path.endswith("/folder"):
            try:
                parts = path.strip("/").split("/")
                body = self.read_json()
                chat = set_chat_folder(parts[2], body.get("folder_id"))
                return self.send_json({"ok": True, "chat": chat})
            except Exception as e:
                return self.send_json({"error": str(e)}, 400)

        if path.startswith("/api/chats/") and path.endswith("/rename"):
            try:
                parts = path.strip("/").split("/")
                if len(parts) != 4:
                    return self.send_json({"error": "Rota inválida"}, 404)
                body = self.read_json()
                chat = rename_chat(parts[2], body.get("title", ""))
                return self.send_json({"ok": True, "chat": chat})
            except FileNotFoundError as e:
                return self.send_json({"error": str(e)}, 404)
            except Exception as e:
                return self.send_json({"error": str(e)}, 400)

        if path.startswith("/api/chats/") and path.endswith("/pin"):
            try:
                parts = path.strip("/").split("/")
                if len(parts) != 4:
                    return self.send_json({"error": "Rota inválida"}, 404)
                body = self.read_json()
                chat = set_chat_pinned(parts[2], bool(body.get("pinned", True)))
                return self.send_json({"ok": True, "chat": chat})
            except FileNotFoundError as e:
                return self.send_json({"error": str(e)}, 404)
            except Exception as e:
                return self.send_json({"error": str(e)}, 400)

        if path == "/api/models/load":
            try:
                body = self.read_json()
                model_id = str(body.get("model_id", ""))
                if not model_id:
                    return self.send_json({"error": "Escolha um modelo."}, 400)
                name = start_model(model_id)
                return self.send_json({"ok": True, "model": name, "status": server_status()})
            except Exception as e:
                return self.send_json({"error": str(e)}, 500)

        if path == "/api/chats":
            return self.send_json(new_chat(), 201)

        if path == "/api/trash/restore":
            try:
                body = self.read_json()
                chat = restore_trash(str(body.get("trash_file", "")))
                return self.send_json({"ok": True, "chat": chat})
            except Exception as e:
                return self.send_json({"error": str(e)}, 400)

        if path.startswith("/api/chats/") and path.endswith("/stream"):
            parts = path.strip("/").split("/")
            if len(parts) != 4:
                return self.send_json({"error": "Rota inválida"}, 404)

            chat_id = parts[2]
            chat = load_chat(chat_id)
            if chat is None:
                return self.send_json({"error": "Chat não encontrado"}, 404)

            try:
                body = self.read_json()
                content = str(body.get("content", "")).strip()
                attachments = body.get("attachments") or []
                if not content and not attachments:
                    return self.send_json({"error": "Mensagem vazia"}, 400)

                replace_from_index = body.get("replace_from_index")
                if replace_from_index is not None:
                    try:
                        replace_from_index = int(replace_from_index)
                    except Exception:
                        return self.send_json({"error": "Índice de edição inválido."}, 400)

                    messages_now = chat.get("messages", []) or []
                    if replace_from_index < 0 or replace_from_index > len(messages_now):
                        return self.send_json({"error": "Índice de edição fora do chat."}, 400)

                    # Editing/regenerating rewinds the conversation to just
                    # before the selected user message. The normal stream path
                    # then appends that user message again and produces a fresh
                    # assistant answer.
                    chat["messages"] = messages_now[:replace_from_index]
                    chat["token_usage"] = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                    save_chat(chat)

                # Hora/data é utilitário local: responde ANTES do modelo.
                # Isso também pega frases mistas como:
                # "eae, só na paz? que horas sao?"
                local_time_intent = detect_local_time_intent(
                    content,
                    chat.get("messages", []),
                )
                if local_time_intent:
                    should_generate_title = chat.get("title") in ("Novo chat", "", None)

                    user_message = {"role": "user", "content": content}
                    if attachments:
                        user_message["attachments"] = attachments
                    chat["messages"].append(user_message)

                    answer = local_time_answer(local_time_intent, content)
                    chat["messages"].append({
                        "role": "assistant",
                        "content": answer,
                    })

                    if should_generate_title:
                        chat["title"] = (
                            "Hora atual"
                            if local_time_intent == "time"
                            else "Data atual"
                        )

                    usage = chat.get("token_usage") or {}
                    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                    total_tokens = int(usage.get("total_tokens", 0) or 0)

                    save_chat(chat)

                    self.sse_start()
                    self.sse("meta", {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "memory_changed": False,
                        "memory_items": memory_count(),
                        "web_used": False,
                        "web_results": 0,
                    })
                    self.sse("delta", {
                        "text": answer,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "approx": False,
                    })
                    self.sse("done", {
                        "chat": chat,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "memory_items": memory_count(),
                    })
                    return

                # Visão está desativada por enquanto. Anexos de imagem podem ser
                # salvos/mostrados no chat, mas nunca causam troca automática de modelo.
                status_now = server_status()

                if not status_now.get("loaded"):
                    return self.send_json({"error": "Carregue um modelo primeiro."}, 400)

                # Update durable memory BEFORE answering so a "sempre faça..." instruction
                # already affects this response.
                memory_changed = maybe_update_memory(content)

                should_generate_title = chat.get("title") in ("Novo chat", "", None)
                user_message = {"role": "user", "content": content}
                if attachments:
                    user_message["attachments"] = attachments
                chat["messages"].append(user_message)
                save_chat(chat)

                web_results = []
                web_context = ""
                web_search_attempted = should_search_web(content)

                if web_search_attempted:
                    if is_external_schedule_query(content):
                        web_results = web_search_with_fallbacks(content, max_results=5)
                    else:
                        web_results = web_search(content, max_results=5)

                    web_context = format_web_context(content, web_results)

                schedule_guard = external_schedule_context_guard(
                    content,
                    web_search_attempted,
                    web_results,
                )

                if schedule_guard:
                    web_context = (
                        (web_context + "\n\n" + schedule_guard).strip()
                        if web_context
                        else schedule_guard
                    )

                messages = build_system_messages(
                    chat["messages"],
                    web_context=web_context,
                )
                prompt_tokens = prompt_token_count(messages)

                previous_usage = chat.get("token_usage") or {}
                previous_prompt_tokens = int(previous_usage.get("prompt_tokens", 0) or 0)
                previous_completion_tokens = int(previous_usage.get("completion_tokens", 0) or 0)
                previous_total_tokens = int(previous_usage.get("total_tokens", 0) or 0)

                payload = {
                    "model": "LLaMA_CPP",
                    "messages": messages,
                    "temperature": float(CFG.get("temperature", 0.7)),
                    "max_tokens": int(CFG.get("max_tokens", -1)),
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }

                req = urllib.request.Request(
                    f"{LLAMAFILE_URL}/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": "Bearer no-key",
                    },
                    method="POST",
                )

                self.sse_start()
                self.sse("meta", {
                    "prompt_tokens": previous_prompt_tokens + prompt_tokens,
                    "completion_tokens": previous_completion_tokens,
                    "total_tokens": previous_total_tokens + prompt_tokens,
                    "memory_changed": memory_changed,
                    "memory_items": memory_count(),
                    "web_used": bool(web_results),
                    "web_attempted": bool(web_search_attempted),
                    "web_results": len(web_results),
                })

                full_text = ""
                approx_completion = 0
                final_usage = None

                try:
                    with urllib.request.urlopen(req, timeout=900) as r:
                        for raw in r:
                            line = raw.decode("utf-8", errors="replace").strip()
                            if not line.startswith("data:"):
                                continue

                            data_text = line[5:].strip()
                            if not data_text or data_text == "[DONE]":
                                continue

                            try:
                                item = json.loads(data_text)
                            except Exception:
                                continue

                            usage = item.get("usage")
                            if usage:
                                final_usage = usage

                            choices = item.get("choices") or []
                            if not choices:
                                continue

                            delta = choices[0].get("delta") or {}
                            piece = delta.get("content")
                            if piece:
                                full_text += str(piece)
                                # Streaming chunks are usually token-sized, but not guaranteed.
                                approx_completion += 1
                                self.sse("delta", {
                                    "text": str(piece),
                                    "prompt_tokens": previous_prompt_tokens + prompt_tokens,
                                    "completion_tokens": previous_completion_tokens + approx_completion,
                                    "total_tokens": previous_total_tokens + prompt_tokens + approx_completion,
                                    "approx": True,
                                })

                except Exception as e:
                    self.sse("error", {"error": str(e)})
                    return

                full_text = normalize_ptbr_output(full_text.strip())
                if not full_text:
                    self.sse("error", {"error": "O modelo terminou sem produzir texto."})
                    return

                # Get an exact-ish final completion token count from the model tokenizer.
                exact_completion = tokenize_text(full_text)
                if final_usage:
                    final_prompt = int(final_usage.get("prompt_tokens", prompt_tokens) or prompt_tokens)
                    final_completion = int(final_usage.get("completion_tokens", exact_completion) or exact_completion)
                    final_total = int(final_usage.get("total_tokens", final_prompt + final_completion) or (final_prompt + final_completion))
                else:
                    final_prompt = prompt_tokens
                    final_completion = exact_completion
                    final_total = final_prompt + final_completion

                cumulative_prompt = previous_prompt_tokens + final_prompt
                cumulative_completion = previous_completion_tokens + final_completion
                cumulative_total = previous_total_tokens + final_total

                chat["messages"].append({"role": "assistant", "content": full_text})
                chat["token_usage"] = {
                    "prompt_tokens": cumulative_prompt,
                    "completion_tokens": cumulative_completion,
                    "total_tokens": cumulative_total,
                }

                if should_generate_title:
                    chat["title"] = generate_chat_title(chat["messages"])

                save_chat(chat)

                self.sse("done", {
                    "chat": chat,
                    "prompt_tokens": cumulative_prompt,
                    "completion_tokens": cumulative_completion,
                    "total_tokens": cumulative_total,
                    "memory_items": memory_count(),
                })
                return

            except BrokenPipeError:
                return
            except Exception as e:
                try:
                    self.sse_start()
                    self.sse("error", {"error": str(e)})
                except Exception:
                    pass
                return

        self.send_error(404)

    def do_DELETE(self):
        path = urlparse(self.path).path


        if path.startswith("/api/folders/"):
            folder_id = path.split("/")[-1]
            if delete_folder(folder_id):
                return self.send_json({"ok": True})
            return self.send_json({"error": "Pasta não encontrada"}, 404)

        if path.startswith("/api/trash/"):
            filename = path.split("/")[-1]
            try:
                permanently_delete_trash(filename)
                return self.send_json({"ok": True, "permanent": True})
            except Exception as e:
                return self.send_json({"error": str(e)}, 404)


        if path.startswith("/api/chats/") and path.endswith("/permanent"):
            try:
                parts = path.strip("/").split("/")
                permanently_delete_chat(parts[2])
                return self.send_json({"ok": True, "permanent": True})
            except Exception as e:
                return self.send_json({"error": str(e)}, 404)

        if path.startswith("/api/chats/"):
            chat_id = path.split("/")[-1]
            soft_delete_chat(chat_id)
            return self.send_json({"ok": True, "recoverable": True})

        self.send_error(404)

def serve():
    server = ThreadingHTTPServer(("127.0.0.1", UI_PORT), Handler)
    print("BatataAI v2.3")
    print(f"Persona: {'OK' if load_persona() else 'NÃO ENCONTRADA'}")
    print(f"Memória persistente: {memory_count()} item(ns)")
    print(f"Interface: http://127.0.0.1:{UI_PORT}")
    print("Chats apagados vão para data/trash/ e podem ser restaurados.")
    threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{UI_PORT}")).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        stop_llamafile()

if __name__ == "__main__":
    serve()
