"""
Tests for the core MemoryUnit class.
"""

import pytest
from unittest.mock import Mock, patch

from memory_unit.models.documents import ContextDocument
from memory_unit.models.query import ContextQueryResult
from memory_unit.drive.client import GoogleDriveClient
from memory_unit.core import MemoryUnit, create_memory_unit


class TestMemoryUnitBasic:
    """Basic initialization and state tests."""

    @patch("memory_unit.storage.vector_store.chromadb.PersistentClient")
    def test_initialization_without_token(self, mock_chroma):
        """MemoryUnit can be created without auth token."""
        mock_client = Mock()
        mock_chroma.return_value = mock_client

        memory = MemoryUnit()

        assert memory.auth_token is None
        assert memory.drive_client is None
        assert memory.vector_store is not None  # Created eagerly
        assert memory.is_hydrated is False

    @patch("memory_unit.storage.vector_store.chromadb.PersistentClient")
    def test_create_memory_unit_factory(self, mock_chroma):
        """Factory creates unit ready for hydration."""
        mock_client = Mock()
        mock_chroma.return_value = mock_client

        memory = create_memory_unit(persist_dir="/tmp/test")

        assert memory.auth_token is None
        assert memory.persist_dir == "/tmp/test"

    @patch("memory_unit.storage.vector_store.chromadb.PersistentClient")
    def test_query_not_hydrated(self, mock_chroma):
        """Query before hydration returns helpful message."""
        mock_client = Mock()
        mock_chroma.return_value = mock_client

        memory = MemoryUnit()
        result = memory.query("test query")

        assert isinstance(result, ContextQueryResult)
        assert "not yet hydrated" in result.answer.lower()

    @patch("memory_unit.storage.vector_store.chromadb.PersistentClient")
    def test_get_stats_empty(self, mock_chroma):
        """Stats reflect empty state."""
        mock_client = Mock()
        mock_chroma.return_value = mock_client

        memory = MemoryUnit()
        stats = memory.get_stats()

        assert stats["is_hydrated"] is False
        assert stats["total_documents"] == 0


class TestMemoryUnitHydration:
    """Tests for hydration with ephemeral token."""

    @patch("memory_unit.storage.vector_store.chromadb.PersistentClient")
    @patch.object(GoogleDriveClient, "get_folder_structure")
    @patch.object(GoogleDriveClient, "list_files_in_folder")
    @patch.object(GoogleDriveClient, "get_file_content")
    def test_hydrate_with_ephemeral_token(self, mock_content, mock_list, mock_structure, mock_chroma):
        """Hydrate receives token at call time, not init."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        mock_structure.return_value = {
            "root": {"name": "Workspace", "webViewLink": "..."},
            "user_provided": {"id": "uf", "name": "User Context"},
            "machine_generated": {"id": "mg", "name": "Machine Context"},
            "raw_subfolders": []
        }

        def list_side_effect(folder_id, mime_type=None):
            if folder_id == "uf":
                return [{
                    "id": "f1", "name": "notes.txt", "mimeType": "text/plain",
                    "modifiedTime": "2024-01-01", "webViewLink": "..."
                }]
            elif folder_id == "mg":
                return [{
                    "id": "f2", "name": "prefs.txt", "mimeType": "text/plain",
                    "modifiedTime": "2024-01-02", "webViewLink": "..."
                }]
            return []

        mock_list.side_effect = list_side_effect
        mock_content.side_effect = lambda file_id, mime_type: "Document content"

        # Create unit WITHOUT token
        memory = MemoryUnit(persist_dir="/tmp/test")
        assert memory.auth_token is None

        # Later, extension provides ephemeral token during hydration
        result = memory.hydrate_from_drive(
            root_folder_id="root123",
            auth_token="ephemeral_oauth_token_from_extension"
        )

        assert result["status"] == "success"
        assert memory.auth_token == "ephemeral_oauth_token_from_extension"
        assert memory.is_hydrated is True

    @patch("memory_unit.storage.vector_store.chromadb.PersistentClient")
    def test_hydrate_requires_folder_id(self, mock_chroma):
        """Hydrate requires folder ID."""
        mock_client = Mock()
        mock_chroma.return_value = mock_client

        memory = MemoryUnit()

        with pytest.raises(TypeError):
            # Missing required root_folder_id
            memory.hydrate_from_drive(auth_token="token123")

    @patch("memory_unit.storage.vector_store.chromadb.PersistentClient")
    def test_hydrate_requires_token(self, mock_chroma):
        """Hydrate requires auth token."""
        mock_client = Mock()
        mock_chroma.return_value = mock_client

        memory = MemoryUnit()

        with pytest.raises(TypeError):
            # Missing required auth_token
            memory.hydrate_from_drive(root_folder_id="folder123")

    @patch("memory_unit.storage.vector_store.chromadb.PersistentClient")
    @patch.object(GoogleDriveClient, "get_folder_structure")
    def test_multiple_hydrations_clear_old_data(self, mock_structure, mock_chroma):
        """Re-hydrating clears old data."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        mock_structure.return_value = {
            "root": {"name": "Workspace", "webViewLink": "..."},
            "user_provided": None,
            "machine_generated": None,
            "raw_subfolders": []
        }

        memory = MemoryUnit()

        # First hydration
        result1 = memory.hydrate_from_drive(
            root_folder_id="folder1",
            auth_token="token1"
        )
        assert result1["status"] == "success"

        # Second hydration (with different token)
        result2 = memory.hydrate_from_drive(
            root_folder_id="folder2",
            auth_token="token2"
        )
        assert result2["status"] == "success"
        assert memory.auth_token == "token2"


class TestMemoryUnitErrorHandling:
    """Error handling tests."""

    @patch("memory_unit.storage.vector_store.chromadb.PersistentClient")
    def test_clear_resets_state(self, mock_chroma):
        """Clear resets all state."""
        mock_client = Mock()
        mock_chroma.return_value = mock_client

        memory = MemoryUnit()
        memory.documents = [ContextDocument(content="test", source="s", filename="f")]
        memory.is_hydrated = True

        memory.clear()

        assert memory.is_hydrated is False
        assert len(memory.documents) == 0
