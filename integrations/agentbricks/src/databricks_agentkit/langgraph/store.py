"""LangGraph long-term memory store on Databricks managed memory.

``DatabricksMemoryStore`` is a LangGraph ``BaseStore`` over the Agent Bricks managed memory API,
reached through the AgentKit client. Pass it as ``store=`` to ``create_agent`` or
``StateGraph.compile`` and any LangGraph or LangMem code that uses the store gets memory that
outlives one conversation, with no Postgres connection, credential rotation or pool tuning::

    from databricks_agentkit.langgraph import checkpointer, memory_store, thread_config

    agent = create_agent(
        model, tools=[...], checkpointer=checkpointer(), store=memory_store()
    )
    agent.invoke(inputs, config=thread_config(session_id, actor))

Who the user is. Every entry belongs to an actor. The store reads it per call from the run config
(``configurable["actor_id"]``, which ``thread_config(session_id, actor)`` sets and the checkpointer
already reads), so one store serves many users. An explicit ``actor=`` overrides it. The actor
partitions memories but is not an access-control boundary: the memory store is. Set the actor from
a trusted identity (the signed-in user), never from a model-chosen or user-supplied field, and use
separate stores for separate tenants.

Each item is one cross-session memory (no ``session_id``) at
``{root}/{namespace...}/{key}``, each segment percent-encoded, holding the value as JSON. Entries
outside ``root``, saved under a session, or not holding a JSON object (for example the plain-text
facts ``memory_tools`` writes) are ignored.

Limits:
  - Search ranking is the service's (full-text over each entry's content and description); there
    are no client-side embeddings, so ``index`` configs and per-item ``index`` field lists are
    rejected. One search returns at most 100 entries, so ``offset + limit`` beyond what the service
    returns comes back short.
  - No TTL.
  - ``put`` isn't atomic: concurrent writers to one key can overwrite each other.
  - ``search`` without a query and ``list_namespaces`` list the actor's entries under ``root``
    (one paged listing per call) and filter client-side.
  - Python 3.10: async code can't read the run config, so pass ``actor=`` (a store per request).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import urllib.parse
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Optional

from langgraph.config import get_config
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

from databricks_agentkit.client import AgentKitClient
from databricks_agentkit.runtime.tool_manifest import resolve_memory_store
from databricks_agentkit.runtime.workspace import workspace_client

if TYPE_CHECKING:
    from databricks_agentkit.memory_store import Memory, MemoryStore

logger = logging.getLogger(__name__)

_SEARCH_MAX = 100  # the service returns at most this many entries per search, with no pagination
_SEARCH_HEADROOM = 50  # extra results requested to cover entries dropped by client-side filters
_LIST_PAGE_SIZE = 100


def _encode_segment(segment: str, what: str) -> str:
    if not isinstance(segment, str) or not segment:
        raise ValueError(f"{what} must be a non-empty string, got {segment!r}")
    encoded = urllib.parse.quote(segment, safe="")
    # quote() leaves dots alone; "." and ".." would read as path navigation, so encode them.
    return encoded.replace(".", "%2E") if encoded in (".", "..") else encoded


def _prefix_path(root: str, namespace: tuple[str, ...]) -> str:
    """The listing prefix for ``namespace``: ends in "/" so ("a",) never matches ("ab",)."""
    segments = [_encode_segment(label, "namespace label") for label in namespace]
    return "/".join([root, *segments, ""])


def _item_path(root: str, namespace: tuple[str, ...], key: str) -> str:
    return _prefix_path(root, namespace) + _encode_segment(key, "key")


def _split_path(root: str, path: str) -> Optional[tuple[tuple[str, ...], str]]:
    """``(namespace, key)`` for a path this store wrote, else None."""
    if not path.startswith(root + "/"):
        return None
    segments = path[len(root) + 1 :].split("/")
    if any(not segment for segment in segments):
        return None
    *namespace, key = (urllib.parse.unquote(segment) for segment in segments)
    return tuple(namespace), key


def _value(memory: Memory) -> Optional[dict[str, Any]]:
    try:
        value = json.loads(memory.content or "")
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _compare_values(item_value: Any, filter_value: Any) -> bool:
    """Filter matching with InMemoryStore semantics ($eq/$ne/$gt/$gte/$lt/$lte, nested dicts)."""
    if isinstance(filter_value, dict):
        if any(k.startswith("$") for k in filter_value):
            return all(
                _apply_operator(item_value, op, operand) for op, operand in filter_value.items()
            )
        if not isinstance(item_value, dict):
            return False
        return all(_compare_values(item_value.get(k), v) for k, v in filter_value.items())
    if isinstance(filter_value, (list, tuple)):
        return (
            isinstance(item_value, (list, tuple))
            and len(item_value) == len(filter_value)
            and all(
                _compare_values(iv, fv) for iv, fv in zip(item_value, filter_value, strict=True)
            )
        )
    return item_value == filter_value


def _apply_operator(value: Any, operator: str, operand: Any) -> bool:
    if operator == "$eq":
        return value == operand
    if operator == "$ne":
        return value != operand
    if operator in ("$gt", "$gte", "$lt", "$lte"):
        try:
            left, right = float(value), float(operand)
        except (TypeError, ValueError):
            return False
        return {
            "$gt": left > right,
            "$gte": left >= right,
            "$lt": left < right,
            "$lte": left <= right,
        }[operator]
    raise ValueError(f"Unsupported filter operator: {operator}")


def _matches(value: dict[str, Any], filter_: Optional[dict[str, Any]]) -> bool:
    return not filter_ or all(_compare_values(value.get(k), v) for k, v in filter_.items())


def _matches_condition(namespace: tuple[str, ...], condition: MatchCondition) -> bool:
    pattern = tuple(condition.path)
    if len(namespace) < len(pattern):
        return False
    window = (
        namespace[: len(pattern)]
        if condition.match_type == "prefix"
        else namespace[-len(pattern) :]
    )
    return all(p == "*" or p == label for p, label in zip(pattern, window, strict=True))


class DatabricksMemoryStore(BaseStore):
    """LangGraph ``BaseStore`` on an Agent Bricks managed memory store.

    Args:
        store: Memory store id. Resolves ``store`` → ``AGENT_MEMORY_STORE`` → the
            ``[memory_store]`` binding in agent.toml; with none, the first call raises.
        actor: The identity whose memories to use. Omit it to read
            ``configurable[actor_key]`` from each call's run config (see ``thread_config``).
        actor_key: The run-config key holding the actor.
        strict_user_key: Optional run-config key that must equal the actor when present, eg
            ``"langgraph_user_id"`` for LangMem namespaces templated on the user id.
        root: Path prefix for this store's entries, so they don't mix with other writers.
        client: An AgentKit client; built from the workspace client on first use when omitted.
    """

    # Plain BaseStore on purpose: LangGraph's batched base pools concurrent callers into one
    # batch, where no caller's identity is visible. Each batch() here runs for one actor.
    supports_ttl = False

    def __init__(
        self,
        store: Optional[str] = None,
        *,
        actor: Optional[str] = None,
        actor_key: str = "actor_id",
        strict_user_key: Optional[str] = None,
        root: str = "/langgraph",
        client: Optional[AgentKitClient] = None,
    ) -> None:
        if not root.startswith("/") or root.endswith("/"):
            raise ValueError(f"root must start with '/' and not end with one, got {root!r}")
        self._store = store
        self._actor = actor
        self._actor_key = actor_key
        self._strict_user_key = strict_user_key
        self._root = root
        self._client = client
        self._handle: Optional[MemoryStore] = None
        self._lock = threading.Lock()

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return self._run(list(ops), self._resolve_actor())

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        # Read the run config here, on the event loop, before hopping to a worker thread.
        actor = self._resolve_actor()
        return await asyncio.to_thread(self._run, list(ops), actor)

    def _run(self, ops: list[Op], actor: str) -> list[Result]:
        handle = self._memory_store()
        return [self._apply(handle, actor, op) for op in ops]

    def _resolve_actor(self) -> str:
        try:
            configurable = get_config().get("configurable", {})
        except RuntimeError as e:
            if self._actor is not None:
                return self._actor  # no run config to check strict mode against
            raise ValueError(
                "DatabricksMemoryStore needs an actor: call it inside a LangGraph run with "
                "config=thread_config(session_id, actor), or pass actor=. Async code on "
                "Python 3.10 can't read the run config, so it must pass actor=."
            ) from e
        actor = self._actor if self._actor is not None else configurable.get(self._actor_key)
        if not actor:
            raise ValueError(
                f"DatabricksMemoryStore needs an actor: the run config has no "
                f"{self._actor_key!r}. Use config=thread_config(session_id, actor) or pass actor=."
            )
        if self._strict_user_key is not None:
            user = configurable.get(self._strict_user_key)
            if user is not None and user != actor:
                raise ValueError(
                    f"Run config {self._strict_user_key!r}={user!r} doesn't match actor {actor!r}; "
                    "refusing to read or write another user's memory."
                )
        return actor

    def _memory_store(self) -> MemoryStore:
        # Built on first use and reused: the client resolves workspace auth, and the id-only
        # handle avoids fetching the store before every call.
        with self._lock:
            if self._handle is None:
                store_id = resolve_memory_store(self._store)
                if not store_id:
                    raise ValueError(
                        "No memory store configured: pass store=, set AGENT_MEMORY_STORE, or bind "
                        "one with `agentbricks memory bind`."
                    )
                client = self._client or AgentKitClient(workspace_client())
                self._handle = client.memory_stores._handle(store_id)
            return self._handle

    def _apply(self, handle: MemoryStore, actor: str, op: Op) -> Result:
        if isinstance(op, GetOp):
            return self._get(handle, actor, op)
        if isinstance(op, PutOp):
            return self._put(handle, actor, op)
        if isinstance(op, SearchOp):
            return self._search(handle, actor, op)
        if isinstance(op, ListNamespacesOp):
            return self._list_namespaces(handle, actor, op)
        raise TypeError(f"Unsupported store operation: {type(op).__name__}")

    def _at(self, handle: MemoryStore, actor: str, path: str) -> list[Memory]:
        # The API ignores session_id alongside path, so recheck both.
        listed = handle.list(actor_id=actor, path=path, page_size=_LIST_PAGE_SIZE)
        return [m for m in listed if m.path == path and m.session_id is None]

    def _under(self, handle: MemoryStore, actor: str, prefix: str) -> Iterator[Memory]:
        for memory in handle.list(actor_id=actor, path_prefix=prefix, page_size=_LIST_PAGE_SIZE):
            if memory.session_id is None and memory.path.startswith(prefix):
                yield memory

    def _get(self, handle: MemoryStore, actor: str, op: GetOp) -> Optional[Item]:
        for memory in self._at(handle, actor, _item_path(self._root, op.namespace, op.key)):
            value = _value(memory)
            if value is not None:
                return Item(value=value, key=op.key, namespace=op.namespace, **_times(memory))
        return None

    def _put(self, handle: MemoryStore, actor: str, op: PutOp) -> None:
        if op.ttl is not None:
            raise NotImplementedError("DatabricksMemoryStore doesn't support TTL (ttl=).")
        if isinstance(op.index, list):
            raise NotImplementedError(
                "DatabricksMemoryStore doesn't support per-item index fields: the managed memory "
                "service ranks search results itself, with no client-side embeddings."
            )
        path = _item_path(self._root, op.namespace, op.key)
        if op.value is None:
            for memory in self._at(handle, actor, path):
                memory.delete()
            return
        content = json.dumps(op.value, ensure_ascii=False, separators=(",", ":"))
        handle.put(actor_id=actor, path=path, content=content)

    def _search(self, handle: MemoryStore, actor: str, op: SearchOp) -> list[SearchItem]:
        prefix = _prefix_path(self._root, op.namespace_prefix)
        if op.query:
            limit = min(_SEARCH_MAX, op.offset + op.limit + _SEARCH_HEADROOM)
            found = handle.search(actor_id=actor, query=op.query, limit=limit, path_prefix=prefix)
            candidates = [(r.memory, r.score) for r in found]
        else:
            candidates = [(m, None) for m in self._under(handle, actor, prefix)]
        items = []
        for memory, score in candidates:
            if memory.session_id is not None or not memory.path.startswith(prefix):
                continue
            parts, value = _split_path(self._root, memory.path), _value(memory)
            if parts is None or value is None or not _matches(value, op.filter):
                continue
            namespace, key = parts
            items.append(SearchItem(namespace, key, value, score=score, **_times(memory)))
        return items[op.offset : op.offset + op.limit]

    def _list_namespaces(
        self, handle: MemoryStore, actor: str, op: ListNamespacesOp
    ) -> list[tuple[str, ...]]:
        namespaces = set()
        for memory in self._under(handle, actor, self._root + "/"):
            parts = _split_path(self._root, memory.path)
            if parts is None:
                continue
            namespace = parts[0]
            if op.match_conditions and not all(
                _matches_condition(namespace, c) for c in op.match_conditions
            ):
                continue
            namespaces.add(namespace[: op.max_depth] if op.max_depth is not None else namespace)
        return sorted(namespaces)[op.offset : op.offset + op.limit]


def _times(memory: Memory) -> dict[str, datetime]:
    now = datetime.now(timezone.utc)
    return {"created_at": memory.create_time or now, "updated_at": memory.update_time or now}


def memory_store(
    store: Optional[str] = None, *, actor: Optional[str] = None, **kwargs
) -> BaseStore:
    """The LangGraph long-term memory store: managed memory when a store is configured.

    Resolves ``store`` → ``AGENT_MEMORY_STORE`` → the ``[memory_store]`` binding in agent.toml.
    With none, returns LangGraph's ``InMemoryStore`` and logs a warning: fine for local
    development, but it keeps nothing across restarts and doesn't partition by actor.
    ``actor`` and ``kwargs`` go to :class:`DatabricksMemoryStore`.
    """
    store_id = resolve_memory_store(store)
    if not store_id:
        from langgraph.store.memory import InMemoryStore

        logger.warning(
            "No memory store configured (pass store=, set AGENT_MEMORY_STORE, or run "
            "`agentbricks memory bind`); using an in-process InMemoryStore that keeps nothing "
            "across restarts and doesn't partition memories by actor."
        )
        return InMemoryStore()
    return DatabricksMemoryStore(store_id, actor=actor, **kwargs)
