"""The actor rule shared by the LangGraph memory store, ``thread_config`` and ``memory_tools``."""

from __future__ import annotations

# The LangGraph memory store keeps each user's entries under ``actor + STORE_ACTOR_SUFFIX``. Memory
# pipelines process only actors that have sessions, and no session uses that id, so pipelines
# leave the store's entries alone.
STORE_ACTOR_SUFFIX = ":langgraph"


def check_actor(actor: str) -> str:
    """Return ``actor``, or raise if it ends with the suffix the LangGraph memory store reserves.

    A plain actor with that ending would share another user's store entries, through a session or
    ``memory_tools``, so it's refused wherever an actor comes in.
    """
    if actor.casefold().endswith(STORE_ACTOR_SUFFIX):
        raise ValueError(
            f"Actor ids can't end with {STORE_ACTOR_SUFFIX!r}: the LangGraph memory store keeps "
            f"each user's entries under '<actor>{STORE_ACTOR_SUFFIX}'."
        )
    return actor
