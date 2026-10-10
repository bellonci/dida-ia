import copy
import json
import math
from pathlib import Path
import re
import shutil
import threading
import time
from .storage import write_private_json


class Memory:
    def __init__(self, path: Path):
        self.path = path
        self.data = {"conversas": []}
        self.lock = threading.RLock()
        self.session_start = time.time()
        self.settings_path = path.with_name(path.stem + ".settings.json")
        try:
            settings = json.loads(self.settings_path.read_text(encoding="utf-8"))
            self.enabled = settings.get("enabled") is True
        except FileNotFoundError:
            self.enabled = True
        except (OSError, ValueError, AttributeError):
            self.enabled = False
        self.load()

    def load(self):
        try:
            if self.path.exists():
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(saved, dict) or not isinstance(saved.get("conversas"), list):
                    raise ValueError("Formato de memória inválido")
                self.data = {"conversas": [c for c in saved["conversas"] if isinstance(c, dict)
                    and isinstance(c.get("user"), str) and isinstance(c.get("bot"), str)
                    and isinstance(c.get("ts", 0), (int, float)) and math.isfinite(c.get("ts", 0))
                    and not self._is_noise(c)][-200:]}
        except Exception as e:
            print(f"[memory] não foi possível carregar: {e}")
            if self.path.exists():
                try:
                    shutil.copy2(self.path, self.path.with_suffix(f".invalid-{time.time_ns()}.json"))
                except OSError as backup_error:
                    print(f"[memory] não foi possível preservar a memória inválida: {backup_error}")
            self.data = {"conversas": []}

    @staticmethod
    def _is_noise(c: dict) -> bool:
        user = str(c.get("user", "")).strip().lower()
        bot = str(c.get("bot", "")).strip().lower()
        if user.startswith("# dida"):
            return True
        if bot in {"desculpa, sem resposta agora.", "desculpa, sem resposta agora"}:
            return True
        if bot.startswith(("desculpa, nao consegui gerar uma resposta", "nao consegui analisar a imagem", "não consegui gerar uma resposta", "não consegui analisar a imagem")):
            return True
        meta = c.get("meta") if isinstance(c.get("meta"), dict) else {}
        if meta.get("error") or meta.get("cancelled"):
            return True
        return False

    def save(self):
        with self.lock:
            write_private_json(self.path, self.data)

    def status(self):
        with self.lock:
            return {"count": len(self.data["conversas"]), "enabled": self.enabled}

    def history(self, limit=50):
        with self.lock:
            return copy.deepcopy(self.data["conversas"][-max(1, min(200, limit)):])

    def set_enabled(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError("enabled deve ser verdadeiro ou falso")
        with self.lock:
            write_private_json(self.settings_path, {"enabled": enabled})
            self.enabled = enabled

    def clear(self):
        with self.lock:
            write_private_json(self.path, {"conversas": []})
            self.data = {"conversas": []}
            for backup in self.path.parent.glob(self.path.stem + ".invalid-*.json"):
                backup.unlink(missing_ok=True)
            self.session_start = time.time()

    def add(self, user: str, bot: str, meta: dict = None):
        if self._is_noise({"user": user, "bot": bot, "meta": meta or {}}):
            return
        with self.lock:
            if not self.enabled:
                return
            new = {"conversas": (self.data["conversas"] + [
                {"ts": time.time(), "user": user[:2000], "bot": bot[:2000], "meta": meta or {}}
            ])[-200:]}
            write_private_json(self.path, new)
            self.data = new

    def search(self, query: str, n: int = 3) -> str:
        with self.lock:
            if not self.enabled or not self.data["conversas"]:
                return ""
        stop = set(
            "a ao aos as de da das do dos e em na nas no nos o os para por que um uma uns umas eu "
            "voce me minha meu com como isso isto ai ja se sua seu suas seus ta te foi ser "
            "nao sim ou mas mais muito pode vai vou fala falar aqui ela ele eles elas esse essa este "
            "esta estou era oi oie ola dida diga dizer sabe quem sou tchau obrigado obrigada".split()
        )
        q_words = {w for w in re.findall(r"\w+", query.lower()) if len(w) >= 3 and w not in stop}
        if len(q_words) < 2:
            return ""
        scored = []
        with self.lock:
            if not self.enabled:
                return ""
            snapshot = list(self.data["conversas"][-50:])
        for c in snapshot:
            if c.get("ts", 0) >= self.session_start:
                continue
            if self._is_noise(c):
                continue
            text = (c["user"] + " " + c["bot"]).lower()
            text_words = set(re.findall(r"\w+", text))
            score = len(q_words & text_words)
            if score >= 2:
                scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return ""
        ctx = []
        for _, c in scored[:n]:
            ctx.append(f"Usuario disse: {c['user']}\nVoce respondeu: {c['bot']}")
        return "\n---\n".join(ctx)

    def recent_messages(self, n: int = 4) -> list:
        with self.lock:
            if not self.enabled:
                return []
            recent = [
                c
                for c in self.data["conversas"]
                if c.get("ts", 0) >= self.session_start and not self._is_noise(c)
            ]
        messages = []
        for c in recent[-n:]:
            messages.append({"role": "user", "content": c["user"][:600]})
            messages.append({"role": "assistant", "content": c["bot"][:1000]})
        return messages
