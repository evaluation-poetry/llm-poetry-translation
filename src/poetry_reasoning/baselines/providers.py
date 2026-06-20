from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
import time
from typing import Any

import requests

from .common import assert_no_reference_leak, build_translation_messages, now_iso, prompt_version


@dataclass
class ProviderResult:
    system_id: str
    system_name: str
    record_id: str
    status: str
    output_text: str = ""
    reasoning_content: str = ""
    error: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)
    token_usage: dict[str, Any] = field(default_factory=dict)
    prompt_version: str = ""
    prompt_messages: list[dict[str, str]] = field(default_factory=list)
    latency_seconds: float = 0.0
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_id": self.system_id,
            "system_name": self.system_name,
            "record_id": self.record_id,
            "status": self.status,
            "output_text": self.output_text,
            "reasoning_content": self.reasoning_content,
            "error": self.error,
            "raw_response": self.raw_response,
            "token_usage": self.token_usage,
            "prompt_version": self.prompt_version,
            "prompt_messages": self.prompt_messages,
            "latency_seconds": self.latency_seconds,
            "created_at": self.created_at,
        }


class BaselineProvider:
    system_id = "base"
    system_name = "Base Provider"

    def available(self) -> tuple[bool, str]:
        return True, ""

    def translate(self, record: dict[str, Any]) -> ProviderResult:
        raise NotImplementedError

    def not_available_result(self, record: dict[str, Any], reason: str) -> ProviderResult:
        return ProviderResult(
            system_id=self.system_id,
            system_name=self.system_name,
            record_id=str(record.get("record_id", "")),
            status="not_available",
            error=reason,
        )

    def error_result(self, record: dict[str, Any], error: Exception, latency: float = 0.0) -> ProviderResult:
        return ProviderResult(
            system_id=self.system_id,
            system_name=self.system_name,
            record_id=str(record.get("record_id", "")),
            status="error",
            error=f"{type(error).__name__}: {error}",
            latency_seconds=latency,
        )


@dataclass(frozen=True)
class TransitClient:
    label: str
    base_url: str
    api_key: str


class TransitAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        balance_error: bool = False,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.balance_error = balance_error
        self.retryable = retryable


def _split_env_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]


def _redact_secret(text: str) -> str:
    redacted = text or ""
    for key, value in os.environ.items():
        if ("KEY" in key or "TOKEN" in key or "SECRET" in key or "PASSWORD" in key) and value:
            redacted = redacted.replace(value, "[REDACTED]")
    return redacted


def _is_balance_error(status_code: int | None, text: str) -> bool:
    lowered = (text or "").lower()
    markers = [
        "insufficient balance",
        "insufficient quota",
        "quota exceeded",
        "billing",
        "credit",
        "no balance",
    ]
    return status_code in {402, 403} and any(marker in lowered for marker in markers) or any(
        marker in lowered for marker in markers
    )


def _is_retryable_status(status_code: int | None) -> bool:
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504}


class DeepSeekProvider(BaselineProvider):
    def __init__(
        self,
        system_id: str,
        system_name: str,
        model: str,
        thinking: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.system_id = system_id
        self.system_name = system_name
        self.model = model
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort

    def available(self) -> tuple[bool, str]:
        if not os.getenv("DEEPSEEK_API_KEY"):
            return False, "Missing DEEPSEEK_API_KEY."
        return True, ""

    def translate(self, record: dict[str, Any]) -> ProviderResult:
        api_key = os.environ["DEEPSEEK_API_KEY"]
        url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions")
        current_prompt_version = prompt_version()
        messages = build_translation_messages(record["source_zh"], version=current_prompt_version)
        assert_no_reference_leak(messages, record)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": int(os.getenv("DEEPSEEK_MAX_TOKENS", "4096")),
        }
        if self.thinking:
            payload["thinking"] = {"type": self.thinking}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        start = time.perf_counter()
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        latency = time.perf_counter() - start
        response.raise_for_status()
        data = response.json()
        message = data.get("choices", [{}])[0].get("message", {})
        return ProviderResult(
            system_id=self.system_id,
            system_name=self.system_name,
            record_id=record["record_id"],
            status="ok",
            output_text=(message.get("content") or "").strip(),
            reasoning_content=(message.get("reasoning_content") or "").strip(),
            raw_response=data,
            token_usage=data.get("usage") or {},
            prompt_version=current_prompt_version,
            prompt_messages=messages,
            latency_seconds=latency,
        )


class QwenProvider(BaselineProvider):
    def __init__(self, system_id: str, system_name: str, model: str | None = None, thinking: bool = False) -> None:
        self.system_id = system_id
        self.system_name = system_name
        self.model = model
        self.thinking = thinking

    def available(self) -> tuple[bool, str]:
        if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")):
            return False, "Missing DASHSCOPE_API_KEY or QWEN_API_KEY."
        return True, ""

    def _chat_completions_url(self) -> str:
        base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    def translate(self, record: dict[str, Any]) -> ProviderResult:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.environ["QWEN_API_KEY"]
        model = self.model or os.getenv("QWEN_MODEL", "qwen3.6-plus")
        current_prompt_version = prompt_version()
        messages = build_translation_messages(record["source_zh"], version=current_prompt_version)
        assert_no_reference_leak(messages, record)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "max_tokens": int(os.getenv("QWEN_MAX_TOKENS", "8192")),
            "enable_thinking": self.thinking,
        }
        if os.getenv("QWEN_THINKING_BUDGET") and self.thinking:
            payload["thinking_budget"] = int(os.environ["QWEN_THINKING_BUDGET"])
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        start = time.perf_counter()
        response = requests.post(self._chat_completions_url(), headers=headers, json=payload, timeout=600)
        latency = time.perf_counter() - start
        response.raise_for_status()
        data = response.json()
        message = data.get("choices", [{}])[0].get("message", {})
        return ProviderResult(
            system_id=self.system_id,
            system_name=self.system_name,
            record_id=record["record_id"],
            status="ok",
            output_text=(message.get("content") or "").strip(),
            reasoning_content=(message.get("reasoning_content") or "").strip(),
            raw_response=data,
            token_usage=data.get("usage") or {},
            prompt_version=current_prompt_version,
            prompt_messages=messages,
            latency_seconds=latency,
        )


class ClaudeMessagesProvider(BaselineProvider):
    def __init__(
        self,
        system_id: str,
        system_name: str,
        model: str = "claude-sonnet-4-6",
        thinking: bool = False,
    ) -> None:
        self.system_id = system_id
        self.system_name = system_name
        self.model = model
        self.thinking = thinking

    def available(self) -> tuple[bool, str]:
        if not self._clients():
            return (
                False,
                "Missing Claude transit credentials. Set a Claude transit API key and base URL.",
            )
        return True, ""

    def _clients(self) -> list[TransitClient]:
        clients: list[TransitClient] = []

        keys = _split_env_list(os.getenv("CLAUDE_TRANSIT_API_KEYS"))
        urls = _split_env_list(os.getenv("CLAUDE_TRANSIT_BASE_URLS"))
        for index, api_key in enumerate(keys):
            base_url = urls[index] if index < len(urls) else os.getenv("CLAUDE_TRANSIT_BASE_URL", "")
            if base_url:
                clients.append(TransitClient(f"list-{index + 1}", base_url, api_key))

        primary_key = os.getenv("CLAUDE_TRANSIT_PRIMARY_API_KEY")
        primary_base_url = os.getenv("CLAUDE_TRANSIT_PRIMARY_BASE_URL", "")
        if primary_key and primary_base_url:
            clients.append(TransitClient("primary", primary_base_url, primary_key))
        secondary_key = os.getenv("CLAUDE_TRANSIT_SECONDARY_API_KEY")
        secondary_base_url = os.getenv("CLAUDE_TRANSIT_SECONDARY_BASE_URL", "")
        if secondary_key and secondary_base_url:
            clients.append(TransitClient("secondary", secondary_base_url, secondary_key))

        single_key = os.getenv("CLAUDE_TRANSIT_API_KEY")
        single_base_url = os.getenv("CLAUDE_TRANSIT_BASE_URL", "")
        if single_key and single_base_url:
            clients.append(TransitClient("single", single_base_url, single_key))

        deduped: list[TransitClient] = []
        seen: set[tuple[str, str]] = set()
        for client in clients:
            key = (client.label, client.base_url.rstrip("/"))
            if key not in seen:
                deduped.append(client)
                seen.add(key)
        return deduped

    @staticmethod
    def _messages_url(base_url: str) -> str:
        clean = base_url.rstrip("/")
        if clean.endswith("/v1/messages"):
            return clean
        if clean.endswith("/messages"):
            return clean
        if clean.endswith("/v1"):
            return f"{clean}/messages"
        return f"{clean}/v1/messages"

    @staticmethod
    def _to_anthropic_messages(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, Any]]]:
        system_parts: list[str] = []
        anthropic_messages: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                system_parts.append(content)
                continue
            if role not in {"user", "assistant"}:
                role = "user"
            anthropic_messages.append({"role": role, "content": content})
        return "\n\n".join(system_parts), anthropic_messages

    @staticmethod
    def _parse_content_blocks(data: dict[str, Any]) -> tuple[str, str]:
        content = data.get("content", [])
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        if isinstance(content, str):
            return content.strip(), ""
        if not isinstance(content, list):
            return "", ""
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(str(block.get("text") or ""))
            elif block_type == "thinking":
                thinking_parts.append(str(block.get("thinking") or block.get("text") or ""))
            elif block_type == "redacted_thinking":
                thinking_parts.append("[redacted_thinking]")
        return "".join(text_parts).strip(), "\n".join(part.strip() for part in thinking_parts if part.strip())

    def _post_once(self, client: TransitClient, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "x-api-key": client.api_key,
            "anthropic-version": os.getenv("CLAUDE_ANTHROPIC_VERSION", "2023-06-01"),
            "content-type": "application/json",
        }
        response = requests.post(
            self._messages_url(client.base_url),
            headers=headers,
            json=payload,
            timeout=float(os.getenv("CLAUDE_TIMEOUT_SECONDS", "600")),
        )
        if response.status_code >= 400:
            body = _redact_secret(response.text[:1000])
            raise TransitAPIError(
                f"Claude transit HTTP {response.status_code}: {body}",
                status_code=response.status_code,
                balance_error=_is_balance_error(response.status_code, body),
                retryable=_is_retryable_status(response.status_code),
            )
        return response.json()

    def _post_with_retries(self, client: TransitClient, payload: dict[str, Any]) -> dict[str, Any]:
        retries = int(os.getenv("CLAUDE_MAX_RETRIES", "3"))
        base_sleep = float(os.getenv("CLAUDE_RETRY_BASE_SECONDS", "2"))
        for attempt in range(retries + 1):
            try:
                return self._post_once(client, payload)
            except requests.RequestException as exc:
                if attempt >= retries:
                    raise TransitAPIError(
                        f"Claude transit request failed: {_redact_secret(str(exc))}",
                        retryable=True,
                    ) from exc
                time.sleep(base_sleep * (2 ** attempt))
            except TransitAPIError as exc:
                if exc.balance_error or not exc.retryable or attempt >= retries:
                    raise
                time.sleep(base_sleep * (2 ** attempt))
        raise TransitAPIError("Claude transit request failed after retries.", retryable=True)

    def translate(self, record: dict[str, Any]) -> ProviderResult:
        current_prompt_version = prompt_version()
        messages = build_translation_messages(record["source_zh"], version=current_prompt_version)
        assert_no_reference_leak(messages, record)
        system, anthropic_messages = self._to_anthropic_messages(messages)
        payload: dict[str, Any] = {
            "model": os.getenv("CLAUDE_MODEL", self.model),
            "system": system,
            "messages": anthropic_messages,
            "max_tokens": int(os.getenv("CLAUDE_MAX_TOKENS", "8192")),
        }
        if self.thinking:
            thinking_payload: dict[str, Any] = {
                "type": "enabled",
                "budget_tokens": int(os.getenv("CLAUDE_THINKING_BUDGET_TOKENS", "1024")),
            }
            if os.getenv("CLAUDE_THINKING_DISPLAY"):
                thinking_payload["display"] = os.environ["CLAUDE_THINKING_DISPLAY"]
            payload["thinking"] = thinking_payload

        start = time.perf_counter()
        errors: list[str] = []
        selected_client = ""
        data: dict[str, Any] | None = None
        for client in self._clients():
            selected_client = client.label
            try:
                data = self._post_with_retries(client, payload)
                break
            except TransitAPIError as exc:
                errors.append(f"{client.label}: {exc}")
                if not (exc.balance_error or exc.retryable):
                    break
                continue
        latency = time.perf_counter() - start
        if data is None:
            raise RuntimeError("; ".join(errors) or "Claude transit request failed.")
        output_text, reasoning_content = self._parse_content_blocks(data)
        raw_response = {**data, "transit_client_label": selected_client}
        token_usage = data.get("usage") or {}
        if self.thinking:
            token_usage = {**token_usage, "thinking_requested": True, "has_thinking_block": bool(reasoning_content)}
        else:
            token_usage = {**token_usage, "thinking_requested": False, "has_thinking_block": bool(reasoning_content)}
        return ProviderResult(
            system_id=self.system_id,
            system_name=self.system_name,
            record_id=record["record_id"],
            status="ok",
            output_text=output_text,
            reasoning_content=reasoning_content,
            raw_response=raw_response,
            token_usage=token_usage,
            prompt_version=current_prompt_version,
            prompt_messages=messages,
            latency_seconds=latency,
        )


class BaiduTranslateProvider(BaselineProvider):
    system_id = "baidu_translate"
    system_name = "Baidu Translate"

    def available(self) -> tuple[bool, str]:
        if os.getenv("BAIDU_TRANSLATE_APP_ID") and os.getenv("BAIDU_TRANSLATE_SECRET_KEY"):
            return True, ""
        return False, "Missing BAIDU_TRANSLATE_APP_ID or BAIDU_TRANSLATE_SECRET_KEY."

    def translate(self, record: dict[str, Any]) -> ProviderResult:
        app_id = os.environ["BAIDU_TRANSLATE_APP_ID"]
        secret = os.environ["BAIDU_TRANSLATE_SECRET_KEY"]
        salt = str(int(time.time() * 1000))
        query = record["source_zh"]
        sign = hashlib.md5(f"{app_id}{query}{salt}{secret}".encode("utf-8")).hexdigest()
        payload = {"q": query, "from": "zh", "to": "en", "appid": app_id, "salt": salt, "sign": sign}
        start = time.perf_counter()
        response = requests.post("https://fanyi-api.baidu.com/api/trans/vip/translate", data=payload, timeout=120)
        latency = time.perf_counter() - start
        response.raise_for_status()
        data = response.json()
        if "error_code" in data:
            raise RuntimeError(f"Baidu error {data.get('error_code')}: {data.get('error_msg')}")
        output = "\n".join(item.get("dst", "") for item in data.get("trans_result", []))
        return ProviderResult(
            system_id=self.system_id,
            system_name=self.system_name,
            record_id=record["record_id"],
            status="ok",
            output_text=output.strip(),
            raw_response=data,
            latency_seconds=latency,
        )


def default_providers() -> list[BaselineProvider]:
    return [
        BaiduTranslateProvider(),
        DeepSeekProvider(
            "deepseek_v4_flash_non_thinking",
            "DeepSeek V4 Flash (non-thinking)",
            "deepseek-v4-flash",
            thinking="disabled",
        ),
        DeepSeekProvider(
            "deepseek_v4_flash_thinking",
            "DeepSeek V4 Flash (thinking)",
            "deepseek-v4-flash",
            thinking="enabled",
            reasoning_effort="high",
        ),
        QwenProvider("qwen36_plus_non_thinking", "Qwen3.6 Plus (non-thinking)", thinking=False),
        QwenProvider("qwen36_plus_thinking", "Qwen3.6 Plus (thinking)", thinking=True),
        ClaudeMessagesProvider(
            "claude_sonnet46_non_thinking",
            "Claude Sonnet 4.6 (non-thinking)",
            "claude-sonnet-4-6",
            thinking=False,
        ),
        ClaudeMessagesProvider(
            "claude_sonnet46_thinking",
            "Claude Sonnet 4.6 (thinking)",
            "claude-sonnet-4-6",
            thinking=True,
        ),
    ]


def full397_provider_ids() -> list[str]:
    return default_provider_ids()


def claude_provider_ids() -> list[str]:
    return [
        "claude_sonnet46_non_thinking",
        "claude_sonnet46_thinking",
    ]


def model_family_provider_ids() -> list[str]:
    return [
        "deepseek_v4_flash_non_thinking",
        "deepseek_v4_flash_thinking",
        "qwen36_plus_non_thinking",
        "qwen36_plus_thinking",
        "claude_sonnet46_non_thinking",
        "claude_sonnet46_thinking",
    ]


def provider_by_id(include_optional: bool = True) -> dict[str, BaselineProvider]:
    del include_optional
    return {provider.system_id: provider for provider in default_providers()}


def default_provider_ids() -> list[str]:
    return [provider.system_id for provider in default_providers()]
