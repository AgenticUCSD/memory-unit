"""
Google Drive client for file operations.
"""

from typing import Dict, Any, List, Optional

import requests


class GoogleDriveClient:
    """
    Client for Google Drive operations.
    Uses OAuth token for authentication.
    """

    def __init__(self, auth_token: str):
        self.auth_token = auth_token
        self.base_url = "https://www.googleapis.com/drive/v3"
        self.upload_url = "https://www.googleapis.com/upload/drive/v3"

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json"
        }

    def list_files_in_folder(self, folder_id: str, mime_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all files in a Drive folder."""
        query = f"'{folder_id}' in parents and trashed = false"
        if mime_type:
            query += f" and mimeType = '{mime_type}'"

        files = []
        page_token = None

        while True:
            params = {
                "q": query,
                "fields": "files(id,name,mimeType,webViewLink,appProperties,modifiedTime)",
                "pageSize": 100
            }
            if page_token:
                params["pageToken"] = page_token

            response = requests.get(
                f"{self.base_url}/files",
                headers=self._get_headers(),
                params=params
            )
            response.raise_for_status()
            data = response.json()

            files.extend(data.get("files", []))
            page_token = data.get("nextPageToken")

            if not page_token:
                break

        return files

    def download_file(self, file_id: str, mime_type: Optional[str] = None) -> bytes:
        """Download a file from Drive."""
        # For Google Workspace files, export to PDF or text
        if mime_type and mime_type.startswith("application/vnd.google-apps"):
            if mime_type == "application/vnd.google-apps.document":
                export_url = f"{self.base_url}/files/{file_id}/export?mimeType=text/plain"
            elif mime_type == "application/vnd.google-apps.spreadsheet":
                export_url = f"{self.base_url}/files/{file_id}/export?mimeType=text/csv"
            else:
                export_url = f"{self.base_url}/files/{file_id}/export?mimeType=application/pdf"

            response = requests.get(export_url, headers=self._get_headers())
        else:
            # Binary download for regular files
            response = requests.get(
                f"{self.base_url}/files/{file_id}?alt=media",
                headers=self._get_headers()
            )

        response.raise_for_status()
        return response.content

    def get_folder_structure(self, root_folder_id: str) -> Dict[str, Any]:
        """
        Get the folder structure with 2 subfolders.
        Expected structure:
        - Root Folder
          - User Provided Context
          - Machine Generated Context
        """
        # Get root folder info
        response = requests.get(
            f"{self.base_url}/files/{root_folder_id}",
            headers=self._get_headers(),
            params={"fields": "id,name,webViewLink"}
        )
        response.raise_for_status()
        root_info = response.json()

        # List subfolders
        subfolders = self.list_files_in_folder(
            root_folder_id,
            mime_type="application/vnd.google-apps.folder"
        )

        user_provided = None
        machine_generated = None

        for folder in subfolders:
            name_lower = folder["name"].lower()
            if "user" in name_lower or "provided" in name_lower or "my knowledge" in name_lower:
                user_provided = folder
            elif "machine" in name_lower or "generated" in name_lower:
                machine_generated = folder

        return {
            "root": root_info,
            "user_provided": user_provided,
            "machine_generated": machine_generated,
            "raw_subfolders": subfolders
        }

    def get_file_content(self, file_id: str, mime_type: str) -> str:
        """Get text content from a Drive file."""
        content = self.download_file(file_id, mime_type)

        if mime_type == "application/pdf":
            return self._extract_pdf_text(content)
        elif mime_type in ["text/plain", "text/markdown"]:
            return content.decode("utf-8")
        elif mime_type == "text/csv":
            return content.decode("utf-8")
        elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return self._extract_docx_text(content)
        else:
            # Try to decode as text
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError:
                return ""

    def _extract_pdf_text(self, content: bytes) -> str:
        """Extract text from PDF bytes."""
        try:
            from pypdf import PdfReader
            import io

            reader = PdfReader(io.BytesIO(content))
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n\n".join(text_parts)
        except ImportError:
            try:
                from PyPDF2 import PdfReader
                import io

                reader = PdfReader(io.BytesIO(content))
                text_parts = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                return "\n\n".join(text_parts)
            except ImportError:
                return ""
        except Exception:
            return ""

    def _extract_docx_text(self, content: bytes) -> str:
        """Extract text from DOCX bytes."""
        try:
            import docx
            import io

            doc = docx.Document(io.BytesIO(content))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except ImportError:
            return ""
        except Exception:
            return ""

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())
