"""One store, many users: the actor comes from each run's config, through a real LangGraph graph.

The store is built without ``actor=``. Every call reads ``configurable["actor_id"]``, which
``thread_config(session_id, actor)`` sets, so this covers the path apps use in production.
"""

from __future__ import annotations

import asyncio
from typing import TypedDict

import pytest

try:
    from langgraph.config import get_store
    from langgraph.graph import END, START, StateGraph

    from databricks_agentkit.langgraph.session_store import thread_config
    from databricks_agentkit.langgraph.store import DatabricksMemoryStore
except ImportError:
    pytest.skip("requires the [langgraph] extra", allow_module_level=True)

from _langgraph_store_fakes import STORE, create_fake_api_fixture


class State(TypedDict, total=False):
    fact: str
    recalled: list[str]


def _remember_then_recall(state: State) -> State:
    store = get_store()
    if state.get("fact"):
        store.put(("memories",), state["fact"], {"text": state["fact"]})
    return {"recalled": sorted(item.key for item in store.search(("memories",)))}


async def _aremember_then_recall(state: State) -> State:
    store = get_store()
    if state.get("fact"):
        await store.aput(("memories",), state["fact"], {"text": state["fact"]})
    return {"recalled": sorted(item.key for item in await store.asearch(("memories",)))}


def _graph(node, store):
    builder = StateGraph(State)
    builder.add_node("memory", node)
    builder.add_edge(START, "memory")
    builder.add_edge("memory", END)
    return builder.compile(store=store)


@pytest.fixture
def api(monkeypatch):
    return create_fake_api_fixture(monkeypatch)


def _actors(api) -> set[str]:
    return {call["actor_id"] for call in api.put_calls}


def test_two_users_through_one_graph_sync(api):
    graph = _graph(_remember_then_recall, DatabricksMemoryStore(store=STORE))

    graph.invoke({"fact": "prefers-email"}, config=thread_config("s1", "alice"))
    graph.invoke({"fact": "likes-rust"}, config=thread_config("s2", "bob"))

    alice = graph.invoke({}, config=thread_config("s3", "alice"))
    bob = graph.invoke({}, config=thread_config("s4", "bob"))
    assert alice["recalled"] == ["prefers-email"]
    assert bob["recalled"] == ["likes-rust"]
    assert _actors(api) == {"alice", "bob"}


@pytest.mark.asyncio
async def test_two_users_through_one_graph_async(api):
    graph = _graph(_aremember_then_recall, DatabricksMemoryStore(store=STORE))

    await graph.ainvoke({"fact": "prefers-email"}, config=thread_config("s1", "alice"))
    await graph.ainvoke({"fact": "likes-rust"}, config=thread_config("s2", "bob"))

    alice = await graph.ainvoke({}, config=thread_config("s3", "alice"))
    bob = await graph.ainvoke({}, config=thread_config("s4", "bob"))
    assert alice["recalled"] == ["prefers-email"]
    assert bob["recalled"] == ["likes-rust"]


@pytest.mark.asyncio
async def test_concurrent_runs_never_mix_users(api):
    graph = _graph(_aremember_then_recall, DatabricksMemoryStore(store=STORE))
    users = [f"user{i % 5}" for i in range(50)]

    await asyncio.gather(
        *(
            graph.ainvoke({"fact": f"{user}-fact-{i}"}, config=thread_config(f"s{i}", user))
            for i, user in enumerate(users)
        )
    )

    for call in api.put_calls:
        assert call["path"].split("/")[-1].startswith(call["actor_id"] + "-fact-")
    for user in set(users):
        recalled = (await graph.ainvoke({}, config=thread_config("check", user)))["recalled"]
        assert recalled and all(key.startswith(user + "-fact-") for key in recalled)


def test_strict_mode_reads_the_run_config(api):
    graph = _graph(
        _remember_then_recall,
        DatabricksMemoryStore(store=STORE, strict_user_key="langgraph_user_id"),
    )
    config = thread_config("s1", "alice")

    config["configurable"]["langgraph_user_id"] = "alice"
    assert graph.invoke({"fact": "ok"}, config=config)["recalled"] == ["ok"]

    config["configurable"]["langgraph_user_id"] = "bob"
    with pytest.raises(ValueError, match="doesn't match actor"):
        graph.invoke({"fact": "leak"}, config=config)
    assert "leak" not in {call["path"].split("/")[-1] for call in api.put_calls}


def test_missing_actor_in_run_config_is_an_error(api):
    graph = _graph(_remember_then_recall, DatabricksMemoryStore(store=STORE))

    with pytest.raises(ValueError, match="thread_config"):
        graph.invoke({"fact": "x"}, config={"configurable": {"thread_id": "s1"}})
    assert api.put_calls == []


def test_outside_a_run_without_actor_is_an_error(api):
    with pytest.raises(ValueError, match="actor="):
        DatabricksMemoryStore(store=STORE).put(("memories",), "k", {"v": 1})
