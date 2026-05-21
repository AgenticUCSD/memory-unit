"""
Tests for Google Drive client.
"""

import pytest
from unittest.mock import Mock, patch

from memory_unit.drive.client import GoogleDriveClient


class TestGoogleDriveClient:
    """Tests for Google Drive client with mocked HTTP calls."""

    def test_initialization(self):
        client = GoogleDriveClient("my_token")
        assert client.auth_token == "my_token"
        headers = client._get_headers()
        assert headers["Authorization"] == "Bearer my_token"

    def test_list_files(self):
        with patch("memory_unit.drive.client.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "files": [
                    {"id": "file1", "name": "doc.pdf", "mimeType": "application/pdf"},
                    {"id": "file2", "name": "notes.txt", "mimeType": "text/plain"}
                ],
                "nextPageToken": None
            }
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            client = GoogleDriveClient("fake_token")
            files = client.list_files_in_folder("folder123")

            assert len(files) == 2
            assert files[0]["name"] == "doc.pdf"

    def test_download_file(self):
        with patch("memory_unit.drive.client.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.content = b"file content here"
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            client = GoogleDriveClient("token")
            content = client.download_file("file123")

            assert content == b"file content here"

    def test_get_folder_structure(self):
        with patch("memory_unit.drive.client.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.side_effect = [
                {"id": "root123", "name": "Root", "webViewLink": "..."},
                {"files": [
                    {"id": "uf", "name": "User Provided Context", "mimeType": "application/vnd.google-apps.folder"},
                    {"id": "mg", "name": "Machine Generated Context", "mimeType": "application/vnd.google-apps.folder"},
                ], "nextPageToken": None}
            ]
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            client = GoogleDriveClient("token")
            structure = client.get_folder_structure("root123")

            assert "root" in structure
            assert "user_provided" in structure
            assert "machine_generated" in structure
            assert structure["user_provided"]["id"] == "uf"

    def test_get_file_content_text(self):
        client = GoogleDriveClient("token")

        with patch.object(client, "download_file", return_value=b"Hello World"):
            content = client.get_file_content("file1", "text/plain")
            assert content == "Hello World"

    def test_get_file_content_pdf(self):
        client = GoogleDriveClient("token")

        with patch.object(client, "download_file", return_value=b"PDF_BYTES"):
            with patch.object(client, "_extract_pdf_text", return_value="Extracted text"):
                content = client.get_file_content("file1", "application/pdf")
                assert content == "Extracted text"

    def test_api_error(self):
        with patch("memory_unit.drive.client.requests.get") as mock_get:
            mock_get.side_effect = Exception("API Error: 401 Unauthorized")

            client = GoogleDriveClient("bad_token")

            with pytest.raises(Exception):
                client.list_files_in_folder("folder123")
