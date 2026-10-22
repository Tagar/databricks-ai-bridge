"""User-Agent tagging for AgentKit-constructed clients.

This module tags WorkspaceClients built by AgentKit itself, not clients passed in from
customers or from Apps. The tag enables telemetry to distinguish AgentKit usage patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient


def _agentkit_version() -> str:
    """Return the AgentKit version string."""
    try:
        from importlib.metadata import version

        return version("databricks-agentbricks")
    except Exception:  # noqa: BLE001 - fallback if metadata unavailable
        return "unknown"


def _tag_workspace_client(client: WorkspaceClient, component: str = "store") -> WorkspaceClient:
    """Tag an AgentKit-constructed WorkspaceClient with user-agent extra.

    Args:
        client: A WorkspaceClient constructed by AgentKit.
        component: The AgentKit component using the client (e.g., "store", "memory_tools").

    Returns:
        The same client, tagged with the user-agent extra.

    Note:
        This function is only called on clients AgentKit constructs, never on clients
        passed in from customers or Apps. If called on a customer's client, it would
        mutate it, which violates the API contract. See #776.
    """
    user_agent_extra = f"agentkit/{_agentkit_version()}/component/{component}"
    try:
        # Try the with_user_agent_extra method if available (newer SDK versions)
        if hasattr(client, "with_user_agent_extra"):
            return client.with_user_agent_extra(user_agent_extra)
    except Exception:  # noqa: BLE001 - fallback to direct header modification
        pass

    # Fallback: set the user-agent extra as a custom header
    # This modifies the client's internal state, which is why we only do this for
    # clients AgentKit constructs, not for reused customers' clients.
    try:
        # The header approach: add to the api_client's default headers if available
        if hasattr(client, "api_client") and hasattr(client.api_client, "user_agent"):
            # Append to existing user-agent string
            current = client.api_client.user_agent or ""
            if current:
                client.api_client.user_agent = f"{current} {user_agent_extra}"
            else:
                client.api_client.user_agent = user_agent_extra
    except Exception:  # noqa: BLE001 - silent fallback
        pass

    return client
