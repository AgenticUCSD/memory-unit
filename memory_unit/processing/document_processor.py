"""
Document processing and chunking.
"""

import re
from typing import List

from memory_unit.models.documents import ContextDocument


class DocumentProcessor:
    """Process and chunk documents for indexing."""

    SUPPORTED_MIME_TYPES = {
        "text/plain": ".txt",
        "text/markdown": ".md",
        "application/pdf": ".pdf",
        "text/csv": ".csv",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.google-apps.document": ".gdoc",
        "application/vnd.google-apps.spreadsheet": ".gsheet",
    }

    @staticmethod
    def chunk_text(
        text: str,
        max_chunk_size: int = 1000,
        overlap: int = 100
    ) -> List[str]:
        """
        Chunk text into overlapping segments using paragraph boundaries.
        """
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                words = current_chunk.split()
                overlap_text = " ".join(words[-overlap:]) if len(words) > overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Handle oversized chunks
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > max_chunk_size * 1.5:
                for i in range(0, len(chunk), max_chunk_size - overlap):
                    final_chunks.append(chunk[i:i + max_chunk_size])
            else:
                final_chunks.append(chunk)

        return final_chunks

    def process_drive_files(
        self,
        drive_client,
        folder_id: str,
        folder_name: str
    ) -> List[ContextDocument]:
        """Process all files from a Drive folder into context documents."""
        documents = []

        files = drive_client.list_files_in_folder(folder_id)

        for file_info in files:
            mime_type = file_info.get("mimeType", "")

            # Skip folders
            if mime_type == "application/vnd.google-apps.folder":
                continue

            # Only process supported types
            if mime_type not in self.SUPPORTED_MIME_TYPES:
                # Try to handle shortcuts
                if mime_type == "application/vnd.google-apps.shortcut":
                    continue  # Skip shortcuts for now
                continue

            try:
                content = drive_client.get_file_content(file_info["id"], mime_type)
                if not content or not content.strip():
                    continue

                chunks = self.chunk_text(content)

                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        doc = ContextDocument(
                            content=chunk,
                            source=file_info.get("webViewLink", ""),
                            filename=file_info["name"],
                            doc_type=self.SUPPORTED_MIME_TYPES.get(mime_type, ".txt"),
                            folder=folder_name,
                            chunk_index=i
                        )
                        documents.append(doc)

            except Exception as e:
                print(f"Error processing {file_info['name']}: {e}")
                continue

        return documents

    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())
