"""Unit tests for AgentKit user-agent tagging."""

from __future__ import annotations

from unittest import mock

import pytest


def test_agentkit_version_returns_package_version():
    """Test that the version can be retrieved."""
    from databricks_agentkit._user_agent import _agentkit_version

    version = _agentkit_version()
    assert isinstance(version, str)
    assert len(version) > 0
    # Should be a valid version string like 0.7.0.dev0 or 'unknown' if metadata unavailable
    assert version != ""


def test_tag_workspace_client_with_version():
    """Test that a WorkspaceClient is tagged with the AgentKit version."""
    from databricks_agentkit._user_agent import _tag_workspace_client

    # Create a mock client
    mock_client = mock.MagicMock()
    mock_client.with_user_agent_extra = mock.MagicMock(return_value=mock_client)

    result = _tag_workspace_client(mock_client, component="store")

    # Should call with_user_agent_extra with the right format
    mock_client.with_user_agent_extra.assert_called_once()
    call_args = mock_client.with_user_agent_extra.call_args[0][0]
    assert "agentkit/" in call_args
    assert "component/store" in call_args


def test_tag_workspace_client_with_different_components():
    """Test that different components are tagged correctly."""
    from databricks_agentkit._user_agent import _tag_workspace_client

    mock_client = mock.MagicMock()
    mock_client.with_user_agent_extra = mock.MagicMock(return_value=mock_client)

    # Test store component
    _tag_workspace_client(mock_client, component="store")
    assert "component/store" in mock_client.with_user_agent_extra.call_args[0][0]

    # Test api component
    mock_client.reset_mock()
    mock_client.with_user_agent_extra = mock.MagicMock(return_value=mock_client)
    _tag_workspace_client(mock_client, component="api")
    assert "component/api" in mock_client.with_user_agent_extra.call_args[0][0]

    # Test runtime component
    mock_client.reset_mock()
    mock_client.with_user_agent_extra = mock.MagicMock(return_value=mock_client)
    _tag_workspace_client(mock_client, component="runtime")
    assert "component/runtime" in mock_client.with_user_agent_extra.call_args[0][0]


def test_tag_workspace_client_default_component():
    """Test that the default component is 'store'."""
    from databricks_agentkit._user_agent import _tag_workspace_client

    mock_client = mock.MagicMock()
    mock_client.with_user_agent_extra = mock.MagicMock(return_value=mock_client)

    _tag_workspace_client(mock_client)

    # Should use 'store' as the default component
    call_args = mock_client.with_user_agent_extra.call_args[0][0]
    assert "component/store" in call_args


def test_tag_workspace_client_fallback_to_api_client_user_agent():
    """Test fallback when with_user_agent_extra is not available."""
    from databricks_agentkit._user_agent import _tag_workspace_client

    # Create a mock client without with_user_agent_extra
    mock_client = mock.MagicMock(spec=[])
    mock_client.api_client = mock.MagicMock()
    mock_client.api_client.user_agent = "existing-agent"

    result = _tag_workspace_client(mock_client, component="store")

    # Should have modified the api_client's user_agent
    assert "agentkit/" in mock_client.api_client.user_agent
    assert "component/store" in mock_client.api_client.user_agent


def test_tag_workspace_client_returns_client():
    """Test that the function returns the same client object."""
    from databricks_agentkit._user_agent import _tag_workspace_client

    mock_client = mock.MagicMock()
    mock_client.with_user_agent_extra = mock.MagicMock(return_value=mock_client)

    result = _tag_workspace_client(mock_client)

    assert result is mock_client


def test_workspace_client_is_tagged():
    """Test that workspace_client() returns a tagged client."""
    from databricks_agentkit.runtime import workspace as workspace_module

    mock_client = mock.MagicMock()
    mock_client.with_user_agent_extra = mock.MagicMock(return_value=mock_client)

    with mock.patch.object(workspace_module, "WorkspaceClient", return_value=mock_client):
        result = workspace_module.workspace_client()

        # Should have called with_user_agent_extra
        mock_client.with_user_agent_extra.assert_called_once()
        call_args = mock_client.with_user_agent_extra.call_args[0][0]
        assert "agentkit/" in call_args
        assert "component/runtime" in call_args


@pytest.mark.parametrize("workspace_id", ["ws-12345", ""])
def test_workspace_client_tagging_with_custom_headers(workspace_id):
    """Test that workspace_client() tags clients even with custom headers."""
    from databricks_agentkit.runtime import workspace as workspace_module

    mock_client = mock.MagicMock()
    mock_client.with_user_agent_extra = mock.MagicMock(return_value=mock_client)

    with mock.patch.dict("os.environ", {"DATABRICKS_WORKSPACE_ID": workspace_id}):
        with mock.patch.object(workspace_module, "WorkspaceClient", return_value=mock_client):
            result = workspace_module.workspace_client()

            # Should have tagged the client
            mock_client.with_user_agent_extra.assert_called_once()
