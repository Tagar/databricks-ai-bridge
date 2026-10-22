"""Shared fake transport layer for DatabricksMemoryStore tests.

Fakes the managed memory API at the transport level, allowing tests to exercise
the real DatabricksMemoryStore, handle resolution, and lock code without network calls.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from databricks_agentkit.memory_store import Memory, MemorySearchResult, MemoryStore

STORE = "langgraph-store"


class FakeMemoryApi:
    """In-memory stand-in for the managed memory entries API.

    Implements the same interface as _AgentBricksApiClient for memory operations,
    keyed exactly like the server API responses.
    """

    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}
        self.clients_built: list[object] = []
        self._next_id = 1
        # Track calls for verification of per-call actor resolution
        self.put_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.list_calls: list[dict[str, Any]] = []

    def seed(
        self,
        path: str,
        content: Optional[str],
        *,
        actor_id: str = "alice",
        session_id: Optional[str] = None,
    ) -> str:
        """Store an entry directly (for test setup); returns its resource name."""
        name = f"memory-stores/{STORE}/entries/e{self._next_id}"
        self._next_id += 1
        now = datetime.now(timezone.utc).isoformat()
        entry = {
            "name": name,
            "actor_id": actor_id,
            "path": path,
            "content": content,
            "create_time": now,
            "update_time": now,
        }
        if session_id is not None:
            entry["session_id"] = session_id
        self.entries[name] = entry
        return name

    def create_memory_entry(self, store, actor_id, path, *, content=None, session_id=None, **_):
        """Create or update a memory entry."""
        self.put_calls.append({"actor_id": actor_id, "path": path, "store": store})

        # Check if an entry already exists - if so, update it
        for _name, entry in list(self.entries.items()):
            if (entry["actor_id"], entry.get("session_id"), entry["path"]) == (
                actor_id,
                session_id,
                path,
            ):
                # Update existing entry
                entry["content"] = content
                entry["update_time"] = datetime.now(timezone.utc).isoformat()
                return dict(entry)

        # Create new entry.
        name = self.seed(path, content, actor_id=actor_id, session_id=session_id)
        return dict(self.entries[name])

    def list_memory_entries(
        self, store, actor_id, *, path=None, path_prefix=None, session_id=None, **_
    ):
        """List memory entries, filtering by actor, path, and session."""
        self.list_calls.append(
            {
                "actor_id": actor_id,
                "path": path,
                "path_prefix": path_prefix,
                "session_id": session_id,
                "store": store,
            }
        )

        matches = [
            dict(entry)
            for entry in self.entries.values()
            if entry["actor_id"] == actor_id
            and entry.get("session_id") == session_id
            and (path is None or entry["path"] == path)
            and (path_prefix is None or entry["path"].startswith(path_prefix))
        ]
        return {"managed_memory_entries": matches}

    def delete_memory_entry(self, store, memory_id):
        """Delete a memory entry by ID."""
        name = f"memory-stores/{store}/entries/{memory_id}"
        if name in self.entries:
            del self.entries[name]
        return {}

    def update_memory_entry(self, store, memory_id, *, content=None, **_):
        """Update a memory entry's content."""
        name = f"memory-stores/{store}/entries/{memory_id}"
        if name in self.entries:
            entry = self.entries[name]
            if content is not None:
                entry["content"] = content
                entry["update_time"] = datetime.now(timezone.utc).isoformat()
            return dict(entry)
        return {}

    def search_memory_entries(
        self, store, actor_id, query=None, *, path_prefix=None, limit=None, session_id=None, **_
    ):
        """Search memory entries by keyword query."""
        self.search_calls.append(
            {
                "actor_id": actor_id,
                "query": query,
                "path_prefix": path_prefix,
                "limit": limit,
                "session_id": session_id,
                "store": store,
            }
        )

        results = []
        for entry in self.entries.values():
            if entry["actor_id"] != actor_id:
                continue
            if entry.get("session_id") != session_id:
                continue
            if path_prefix is not None and not entry["path"].startswith(path_prefix):
                continue

            # If query is provided, check for keyword match
            if query:
                content = entry.get("content", "")
                if not isinstance(content, str):
                    content = str(content)

                if query.lower() in content.lower():
                    results.append({"memory": entry, "score": 1.0})
            else:
                # No query: return all matching entries
                results.append({"memory": entry, "score": 1.0})

        # Apply limit
        if limit is not None:
            results = results[:limit]

        return {"results": results}


class FakeMemoryStores:
    """Fake MemoryStores that creates handles backed by our fake API."""

    def __init__(self, fake_api):
        self.fake_api = fake_api

    def _delete_memory(self, store, memory_id):
        """Delete a memory via the API."""
        store_name = store.name.split("/")[-1] if hasattr(store, "name") else STORE
        return self.fake_api.delete_memory_entry(store_name, memory_id)

    def _update_memory(self, store, memory_id, content=None, description=None):
        """Update a memory via the API."""
        store_name = store.name.split("/")[-1] if hasattr(store, "name") else STORE
        return self.fake_api.update_memory_entry(store_name, memory_id, content=content)

    def _handle(self, store_id: str):
        """Return a handle that uses our fake API."""
        # Create a reference to self so FakeHandle can use it as _client for Memory objects
        memory_stores = self

        class FakeHandle:
            def __init__(self, api):
                self.api = api
                self.store_id = store_id

            def put(self, actor_id: str, path: str, content: str, session_id=None):
                self.api.create_memory_entry(
                    self.store_id, actor_id, path, content=content, session_id=session_id
                )

            def list(self, actor_id: str, *, path=None, path_prefix=None, session_id=None, **_):
                response = self.api.list_memory_entries(
                    self.store_id,
                    actor_id,
                    path=path,
                    path_prefix=path_prefix,
                    session_id=session_id,
                )
                # Create a dummy store for constructing Memory objects.
                dummy_store = MemoryStore(
                    name=f"memory-stores/{self.store_id}",
                    display_name="Test Store",
                    _client=memory_stores,
                )
                for entry in response.get("managed_memory_entries", []):
                    yield Memory(
                        name=entry["name"],
                        actor_id=entry["actor_id"],
                        path=entry["path"],
                        session_id=entry.get("session_id"),
                        content=entry.get("content"),
                        create_time=entry.get("create_time"),
                        update_time=entry.get("update_time"),
                        _store=dummy_store,
                        _client=memory_stores,
                    )

            def search(
                self,
                actor_id: str,
                *,
                query=None,
                path_prefix=None,
                limit=None,
                session_id=None,
                **_,
            ):
                response = self.api.search_memory_entries(
                    self.store_id,
                    actor_id,
                    query=query,
                    path_prefix=path_prefix,
                    limit=limit,
                    session_id=session_id,
                )
                dummy_store = MemoryStore(
                    name=f"memory-stores/{self.store_id}",
                    display_name="Test Store",
                    _client=memory_stores,
                )
                for item in response.get("results", []):
                    entry = item["memory"]
                    memory = Memory(
                        name=entry["name"],
                        actor_id=entry["actor_id"],
                        path=entry["path"],
                        session_id=entry.get("session_id"),
                        content=entry.get("content"),
                        create_time=entry.get("create_time"),
                        update_time=entry.get("update_time"),
                        _store=dummy_store,
                        _client=memory_stores,
                    )
                    yield MemorySearchResult(memory=memory, score=item.get("score", 1.0))

        return FakeHandle(self.fake_api)


def create_fake_api_fixture(monkeypatch):
    """Create and wire in a FakeMemoryApi; returns the fake instance."""
    fake = FakeMemoryApi()

    def workspace_client_func():
        return object()  # clients are counted when FakeAgentKitClient is built

    # Patch workspace_client at the location where it's imported/used in store.py
    monkeypatch.setattr(
        "databricks_agentkit.langgraph.store.workspace_client", workspace_client_func
    )

    class FakeAgentKitClient:
        """Fake AgentKitClient that uses our fake API."""

        def __init__(self, workspace_client_param=None):
            fake.clients_built.append(object())
            self.memory_stores = FakeMemoryStores(fake)

    # Patch AgentKitClient at the location where it's used in store.py
    monkeypatch.setattr(
        "databricks_agentkit.langgraph.store.AgentKitClient",
        FakeAgentKitClient,
    )

    return fake
