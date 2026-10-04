"""Адаптер моделей: один интерфейс для языковой модели и модели со зрением.

Сейчас — локальный Ollama. У модели нет инструментов; ответ ограничен JSON-схемой и
дополнительно проверяется кодом. Содержимое документов передаётся только как данные.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_URL = os.environ.get("COURTDOCS_LLM_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("COURTDOCS_LLM_MODEL", "qwen3.5:9b")
DEFAULT_VLM = os.environ.get("COURTDOCS_VLM_MODEL", DEFAULT_MODEL)

MAX_RETRIES = 2  # только для временных сетевых ошибок


class ModelError(RuntimeError):
    pass


@dataclass
class ModelCall:
    model: str
    purpose: str
    seconds: float
    prompt_tokens: int | None
    output_tokens: int | None
    ok: bool
    error: str = ""


@dataclass
class ModelClient:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_URL
    num_ctx: int = 16384
    timeout: int = 600
    calls: list[ModelCall] = field(default_factory=list)

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=5) as r:
                names = [m["name"] for m in json.load(r).get("models", [])]
            return self.model in names
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/api/chat", json.dumps(payload).encode("utf-8"), {"Content-Type": "application/json"}
        )
        last: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.load(r)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")[:300]
                if exc.code < 500 and exc.code != 429:
                    raise ModelError(f"HTTP {exc.code}: {body}") from exc
                last = exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
            time.sleep(1 + attempt)
        raise ModelError(f"модель недоступна: {last}")

    def chat_json(self, system: str, user: str, schema: dict, purpose: str, images: list[bytes] | None = None) -> dict:
        """Один смысловой вызов. Невалидный JSON допускает один запрос исправления формата."""
        message = {"role": "user", "content": user}
        if images:
            message["images"] = [base64.b64encode(i).decode("ascii") for i in images]
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, message],
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"temperature": 0, "seed": 0, "num_ctx": self.num_ctx},
        }
        started = time.time()
        try:
            data = self._post(payload)
            content = data.get("message", {}).get("content", "")
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                payload["messages"] += [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "Ответ не является корректным JSON по схеме. Верни тот же ответ строго в JSON, без новых сведений."},
                ]
                data = self._post(payload)
                result = json.loads(data.get("message", {}).get("content", ""))
        except (ModelError, json.JSONDecodeError) as exc:
            self.calls.append(ModelCall(self.model, purpose, round(time.time() - started, 2), None, None, False, str(exc)[:200]))
            raise ModelError(str(exc)) from exc
        self.calls.append(ModelCall(
            self.model, purpose, round(time.time() - started, 2),
            data.get("prompt_eval_count"), data.get("eval_count"), True,
        ))
        return result
