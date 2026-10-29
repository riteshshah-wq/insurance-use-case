"""GoldenRay MaaS LLM client wrapper.

Works with any model served through GoldenRay MaaS:
- Self-hosted small models: TinyLlama (Ollama), opt-125m (vLLM)
- Self-hosted GPU models: Llama-3, Mixtral, etc.
- Cloud models via External Models gateway: GPT-4o, Claude, etc.

All accessed through the same GoldenRay inference API — just change
the namespace and endpoint in config.
"""

import httpx
import json
import logging
from typing import List, Dict, Optional

from .config import settings

logger = logging.getLogger(__name__)


async def chat(
    messages: List[Dict[str, str]],
    model_endpoint: Optional[str] = None,
    namespace: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 200,
    temperature: float = 0.3,
) -> str:
    """Send a chat completion request to GoldenRay MaaS.

    Args:
        messages: List of {"role": "...", "content": "..."} dicts.
        model_endpoint: Override the default model endpoint name.
        namespace: Override the default namespace/tenant.
        api_key: Override the default API key.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.

    Returns:
        The assistant's response text.
    """
    endpoint = model_endpoint or settings.goldenray_endpoint
    ns = namespace or settings.goldenray_namespace
    key = api_key or settings.goldenray_api_key

    # Primary: GoldenRay MaaS inference API
    url = f"{settings.goldenray_api_url}/api/v1/inference/{ns}/{endpoint}"

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            # Handle OpenAI-compatible response format
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            # Handle raw text response
            if "response" in data:
                return data["response"]
            # Handle Ollama-style response
            if "message" in data:
                return data["message"].get("content", str(data["message"]))

            return str(data)

    except httpx.HTTPError as e:
        logger.warning(f"Primary MaaS API failed: {e}. Trying external URL...")
        return await _fallback_external(messages, ns, endpoint, key, max_tokens, temperature)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise


async def _fallback_external(
    messages: List[Dict[str, str]],
    namespace: str,
    endpoint: str,
    api_key: Optional[str],
    max_tokens: int,
    temperature: float,
) -> str:
    """Fallback to the external MaaS URL."""
    url = (
        f"{settings.goldenray_external_url}/maas/{namespace}/{endpoint}"
        f"/v1/chat/completions"
    )

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        return str(data)


async def generate(
    prompt: str,
    system_prompt: Optional[str] = None,
    **kwargs,
) -> str:
    """Convenience wrapper: build messages from a simple prompt."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return await chat(messages, **kwargs)
