"""Construct Databricks SDK clients with workspace routing when required."""

from __future__ import annotations

import os

from databricks.sdk import WorkspaceClient

from databricks_agentkit._user_agent import _tag_workspace_client


def workspace_headers() -> dict[str, str]:
    """Return headers required to route an account-host request to its workspace."""
    workspace_id = os.getenv("DATABRICKS_WORKSPACE_ID", "").strip()
    return {"X-Databricks-Org-Id": workspace_id} if workspace_id else {}


def workspace_client() -> WorkspaceClient:
    """Return the environment-authenticated client for the active workspace.

    ``databricks apps run-local`` can authenticate through an account-level vanity host while
    exposing the target workspace through ``DATABRICKS_WORKSPACE_ID``. The SDK needs the same
    routing header as the Databricks CLI for those profiles. Ordinary workspace hosts and deployed
    Apps continue to use the SDK's default authentication chain.

    The returned client is tagged with AgentKit user-agent information for telemetry.
    """
    headers = workspace_headers()
    if not headers:
        return _tag_workspace_client(WorkspaceClient(), component="runtime")
    return _tag_workspace_client(WorkspaceClient(custom_headers=headers), component="runtime")
