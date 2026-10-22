# Migrate from DatabricksStore to managed memory

This guide covers migrating an agent using the legacy `DatabricksStore` from `databricks-langchain` to the new managed memory store backed by Databricks Lakebase.

## Why migrate

- **Simpler deployment**: No Postgres connection, no Lakebase project setup. Managed memory stores are provisioned and accessed via the AgentKit REST client, same as sessions.
- **Integrated with Agent Bricks**: Memory is automatically provisioned when you deploy an agent with `agent.toml`, and access is granted to the app automatically.

## Before you start

- Review the limits in the [LangGraph memory store section](README.md#langgraph-memory-store-as-a-basestore) of the README, including the 100-result search cap and last-writer-wins semantics.
- Search results are ranked by the managed-memory service. Client-side embeddings and custom index fields are not supported; if you need those capabilities, keep using `DatabricksStore` with a Lakebase project.
- Existing memories stored with `DatabricksStore` are not migrated; plan a data strategy (export, reimport, or accept data loss).

## Migration steps

### 1. Remove the Postgres connection from your agent code

Remove imports and code that create a Postgres connection or use `DatabricksStore` directly:

```python
# Before:
from databricks_langchain import DatabricksStore
from sqlalchemy.pool import NullPool
import psycopg

conn = psycopg.connect(...)  # or use LakebasePool
store = DatabricksStore(engine=...)

# After:
from databricks_agentkit.langgraph import memory_store

store = memory_store(actor=actor_id)
```

### 2. Update imports and initialization

Replace imports from `databricks-langchain` with imports from `databricks-agentkit`:

```python
# Before:
from databricks_langchain import AsyncDatabricksStore, CheckpointSaver

# After:
from databricks_agentkit.langgraph import memory_store, checkpointer, thread_config
```

### 3. Adapt your graph to use the managed store

Pass the managed memory store to your graph's state, alongside the checkpointer for session persistence:

```python
from databricks_agentkit.langgraph import checkpointer, thread_config, memory_store

# Create the store once and reuse it for each invocation
store = memory_store(actor=actor_id)

# Pass checkpointer for session persistence, your own graph logic for memory access
agent = create_agent(
    model=...,
    tools=[...],
    checkpointer=checkpointer(),  # manages session state
)

# In your invocation loop, pass the actor in the config
config = thread_config(session_id, actor=actor_id)
result = await agent.ainvoke(inputs, config=config)

# Within your graph, read and write memory using store operations
await store.aget(("path", "/key"))
await store.aput(("path", "/key"), {"data": "value"})
```

### 4. Update your agent.toml

Declare the memory store binding so it is provisioned and the app is granted access on deploy:

```toml
[memory_store]
store = "my-agent-memory"
```

Or use `agentbricks memory bind` if you already have a store:

```sh
agentbricks memory bind existing-store-name
```

### 5. Remove custom Postgres/Lakebase setup

If your agent code or deployment script creates a Lakebase project, sets up schemas, or manages Postgres connections, remove that infrastructure. The managed memory store handles provisioning:

```sh
# Before:
agentbricks deploy my-agent --lakebase-project ...

# After:
agentbricks deploy my-agent  # memory_store binding is used automatically
```

### 6. Update your code to pass actor from request context

Ensure your graph receives the actor (typically the signed-in user) from a trusted source, not a user-supplied field:

```python
# Before:
actor = request.json.get("actor")  # ❌ untrusted user input

# After:
from databricks.sdk import WorkspaceClient
client = WorkspaceClient()
actor = client.current_user.me().user_name  # ✅ trusted identity
```

### 7. Test with the new store locally

Run your agent locally with the memory store configured:

```sh
AGENT_MEMORY_STORE=test-memory-store agentbricks dev
```

If memory is not configured, the store falls back to in-memory with a warning, so your agent code continues to work.

### 8. Deploy

Deploy your agent with the updated code and `agent.toml`:

```sh
agentbricks deploy my-agent
```

The managed memory store is provisioned automatically and the app is granted access.

## Data migration considerations

Memories written with `DatabricksStore` are stored in a Lakebase table you own; memories written with the managed memory store are in a Databricks-managed location. They are not compatible:

- **Export old memories**: If you need to preserve existing memories, export them from your `DatabricksStore` table before deployment.
- **Reimport into managed store**: Use `agentbricks memory entries create` or the AgentKit SDK (`memory_store.add()`) to write them.
- **Accept data loss**: If old memories are historical and can be discarded, deploying new code with the managed store starts fresh.

## Rollback

If you need to roll back to `DatabricksStore`, keep a copy of your old code branch and deployment. The legacy package remains available; redeploy from the old branch and re-establish your Postgres/Lakebase connection.
