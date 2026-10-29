"""Customer service chat agent with RAG support."""

import logging
from typing import List, Dict

from ..common.llm_client import chat
from ..common.models import ChatRequest, ChatResponse, ChatMessage
from .rag import retrieve
from .prompts import AGENT_SYSTEM_PROMPT, RAG_QUERY_PROMPT

logger = logging.getLogger(__name__)


async def handle_chat(request: ChatRequest) -> ChatResponse:
    """Process a customer chat message with RAG context.

    1. Retrieve relevant policy documents
    2. Build context-augmented prompt
    3. Generate response via LLM
    4. Return response with source references

    Args:
        request: Chat request with message and history.

    Returns:
        ChatResponse with answer and source references.
    """
    # Retrieve relevant documents
    relevant_chunks = retrieve(request.message, top_k=3)
    sources = list(set(chunk["document"] for chunk in relevant_chunks))

    # Build messages
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT}
    ]

    # Add conversation history
    for msg in request.conversation_history[-6:]:  # Keep last 6 messages
        messages.append({"role": msg.role, "content": msg.content})

    # Build the user message with RAG context
    if relevant_chunks:
        context = "\n\n".join(
            f"[Source: {c['document']}]\n{c['content']}"
            for c in relevant_chunks
        )
        user_prompt = RAG_QUERY_PROMPT.format(
            context=context,
            question=request.message,
        )
    else:
        user_prompt = request.message

    messages.append({"role": "user", "content": user_prompt})

    # Generate response
    try:
        response_text = await chat(messages, max_tokens=300)
    except Exception as e:
        logger.error(f"Agent chat failed: {e}")
        response_text = (
            "I apologize, but I'm having trouble processing your request right now. "
            "Please try again or contact our support team at 1-800-ACME-INS."
        )

    return ChatResponse(
        response=response_text,
        sources=sources,
    )
