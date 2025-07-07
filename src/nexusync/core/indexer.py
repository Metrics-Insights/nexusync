# src/core/indexer.py

import os
from typing import List, Optional, Dict, Any
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from nexusync.utils.logging_config import get_logger
from llama_index.core import Settings
import hashlib

SUPPORTED_DOCUMENTS = [
    ".ics",
    ".csv",
    ".tsv",
    ".doc",
    ".docx",
    ".odt",
    ".epub",
    ".org",
    ".rst",
    ".rtf",
    ".md",
    ".msg",
    ".pdf",
    ".heic",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".bmp",
    ".ppt",
    ".pptx",
    ".xlsx",
    ".eml",
    ".html",
    ".xml",
    ".txt",
    ".json",
]


# --- Helper function for checksum generation ---
def _generate_checksum(
        input_dirs: List[str], recursive: bool, supported_exts: List[str]
) -> str:
    """
    Generates a SHA256 checksum based on the file paths and modification times
    of all supported documents in the input directories.
    """
    hasher = hashlib.sha256()
    file_signatures = []

    for dir_path in input_dirs:
        if not os.path.isdir(dir_path):
            continue  # Skip non-existent directories

        # Use os.walk for recursive search, or os.listdir for non-recursive
        if recursive:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if any(file.endswith(ext) for ext in supported_exts):
                        file_path = os.path.join(root, file)
                        try:
                            mtime = os.path.getmtime(file_path)
                            file_signatures.append(f"{file_path}:{mtime}")
                        except OSError:
                            # File might be deleted between listing and statting
                            continue
        else:
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                if os.path.isfile(file_path) and any(
                        file.endswith(ext) for ext in supported_exts
                ):
                    try:
                        mtime = os.path.getmtime(file_path)
                        file_signatures.append(f"{file_path}:{mtime}")
                    except OSError:
                        continue

    # Sort the signatures to ensure a consistent hash regardless of file system order
    file_signatures.sort()

    # Update the hash with the concatenated string of signatures
    for signature in file_signatures:
        hasher.update(signature.encode('utf-8'))

    return hasher.hexdigest()

class Indexer:
    """
    Indexer is responsible for managing the indexing operations, including creating, refreshing,
    and deleting documents from the index. It supports integration with Chroma for efficient similarity search.

    Attributes:
        input_dirs (List[str]): A list of directory paths containing documents to be indexed.
        recursive (bool): Indicates if subdirectories within input_dirs should be scanned for documents.
        chroma_db_dir (str): The directory where the Chroma database is stored.
        index_persist_dir (str): The directory where the index is persisted to disk for future use.
        chroma_collection_name (str): The name of the collection within the Chroma database.
        index (VectorStoreIndex): The current index instance, loaded or created during initialization.
        logger (logging.Logger): A logger instance for logging operations and errors.
        storage_context (StorageContext): The context for managing the storage and loading of the index.
    """

    def __init__(
        self,
        input_dirs: List[str],
        recursive: bool = True,
        chroma_db_dir: str = "chroma_db",
        index_persist_dir: str = "index_storage",
        chroma_collection_name: str = "my_collection",
        chunk_size: int = 1024,  # Default from llamaindex
        chunk_overlap: int = 20,  # Default from llamaindex
    ):
        """
        Initialize the Indexer with the given parameters.

        Args:
            input_dirs (List[str]): Directories containing documents to be indexed.
            recursive (bool, optional): Scan subdirectories if True. Defaults to True.
            chroma_db_dir (str, optional): Directory for Chroma database. Defaults to "chroma_db".
            index_persist_dir (str, optional): Directory to persist the index. Defaults to "index_storage".
            chroma_collection_name (str, optional): Name of the Chroma collection. Defaults to "my_collection".
            chunk_size (int, optional): Size of each text chunk. Defaults to 1024.
            chunk_overlap (int, optional): Overlap between chunks. Defaults to 20.

        Note:
            The __init__ method doesn't create the index immediately. Instead, it calls the _initiate method,
            which either loads an existing index or builds a new one.
        """
        self.logger = get_logger("nexusync.core.indexer")  # Use full logger name
        self.input_dirs = input_dirs
        self.recursive = recursive
        self.chroma_db_dir = chroma_db_dir
        self.index_persist_dir = index_persist_dir
        self.chroma_collection_name = chroma_collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index = None
        Settings.chunk_overlap = chunk_overlap
        Settings.chunk_size = chunk_size

    def initialize_index(self):
        """
        Load an existing index from storage or create a new one if not found.

        Raises:
            ValueError: If no documents are found in the specified directories.
        """
        checksum_path = os.path.join(self.index_persist_dir, "checksum.txt")

        # Step 1: Generate a checksum of the current state of documents
        self.logger.info("Generating checksum for current documents...")
        new_checksum = _generate_checksum(self.input_dirs, self.recursive, SUPPORTED_DOCUMENTS)
        self.logger.info(f"Current document checksum: {new_checksum}")

        # Step 2: Check if a rebuild is necessary
        old_checksum = None
        if (
                os.path.exists(checksum_path)
                and os.path.exists(self.index_persist_dir)
                and os.path.exists(self.chroma_db_dir)
        ):
            try:
                with open(checksum_path, "r") as f:
                    old_checksum = f.read().strip()
                self.logger.info(f"Found previous checksum: {old_checksum}")
            except IOError as e:
                self.logger.warning(f"Could not read previous checksum file: {e}. Forcing rebuild.")

        if new_checksum and old_checksum == new_checksum:
            self.logger.info("Checksums match. No changes detected in source documents. Skipping index rebuild.")

            self.storage_context = StorageContext.from_defaults(
                persist_dir=self.index_persist_dir
            )
            self.index = load_index_from_storage(self.storage_context)
            self.logger.info("Index already built. Loading from disk.")

            return self.index# Bypass the entire rebuild process

        self.logger.info("Document changes detected or first run. Proceeding with index rebuild.")

        try:
            self.logger.warning("Index not found. Building a new index.")

            self.document_list = []
            total_files = 0
            for file_path in self.input_dirs:
                if not os.path.isdir(file_path):
                    self.logger.error(f"Directory {file_path} does not exist.")
                    raise ValueError(f"Directory {file_path} does not exist.")
                file_count = sum(
                    len(files)
                    for _, _, files in os.walk(file_path)
                    if self.recursive or _ == file_path
                )
                total_files += file_count
                documents = SimpleDirectoryReader(
                    file_path, filename_as_id=True, recursive=self.recursive, required_exts=SUPPORTED_DOCUMENTS
                ).load_data()
                self.logger.info(f"Loaded {len(documents)} files from all directories, under {file_path}")
                self.document_list.extend(documents)

            self.index = VectorStoreIndex.from_documents(self.document_list)
            self.index.storage_context.persist(persist_dir=self.index_persist_dir)
            chroma_client = chromadb.PersistentClient(path=self.chroma_db_dir)
            chroma_collection = chroma_client.get_or_create_collection(
                self.chroma_collection_name
            )
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            self.storage_context = StorageContext.from_defaults(
                persist_dir=self.index_persist_dir, vector_store=vector_store
            )

            if not self.document_list:
                self.logger.error("No documents found to build the index.")
                raise ValueError("No documents found to build the index.")

            self.logger.info("Index Built.")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during initiation: {e}")
            raise
        finally:
            # Save the new checksum after a successful build
            try:
                with open(checksum_path, "w") as f:
                    f.write(new_checksum)
                self.logger.info(f"Successfully built index and saved new checksum to {checksum_path}")
            except IOError as e:
                self.logger.error(f"Failed to save new checksum file: {e}")

        return self.index

    def refresh(self):
        """
        Refresh the index by performing incremental updates and deletions based on the current
        state of the files.

        Raises:
            RuntimeError: If an error occurs during the refresh process.
        """
        self.logger.info("Starting index refresh process...")
        try:
            # Step 1: Collect current files
            current_files = set()
            for input_dir in self.input_dirs:
                for root, _, files in os.walk(input_dir):
                    for file in files:
                        current_files.add(os.path.abspath(os.path.join(root, file)))

            # Step 2: Perform upinsert (this will add new and update existing documents)
            self.upinsert()

            # Step 3: Perform delete (this will remove documents that no longer exist)
            self.delete(current_files)

            # Step 4: Verify and log the results
            updated_stats = self.get_index_stats()

            if updated_stats["num_documents"] != len(current_files):
                self.logger.warning(
                    f"Mismatch between indexed documents ({updated_stats['num_documents']}) and files in directories ({len(current_files)})"
                )

        except Exception as e:
            self.logger.error(
                f"An error occurred during index refresh: {e}", exc_info=True
            )
            raise

    def upinsert(self):
        """
        Upsert (update or insert) documents into the index based on changes or new additions.

        Raises:
            RuntimeError: If an error occurs while performing the upinsert operation.
        """
        total_documents = 0
        total_refreshed = 0

        for input_dir in self.input_dirs:
            self.logger.info(f"Processing directory: {input_dir}")
            documents = SimpleDirectoryReader(
                input_dir, recursive=self.recursive, filename_as_id=True, required_exts=SUPPORTED_DOCUMENTS
            ).load_data()
            total_documents += len(documents)
            loaded_file_count = self.get_index_stats()["num_documents"]
            self.logger.info(f"Loaded {loaded_file_count} files from {input_dir}")

            refreshed_docs = self.index.refresh_ref_docs(documents)
            num_refreshed = sum(1 for r in refreshed_docs if r)
            total_refreshed += num_refreshed

            if num_refreshed == 0:
                self.logger.info(f"No files were modified or added in {input_dir}")
            else:
                for doc, is_refreshed in zip(documents, refreshed_docs):
                    if is_refreshed:
                        doc_path = doc.metadata.get("file_path", "Unknown path")
                        self.logger.info(f"Updated file: {doc_path}")

        if total_refreshed == 0:
            self.logger.info("No files were modified or added in any directory")
        else:
            self.logger.info(f"Total files modified or added: {total_refreshed}")

    def delete(self, current_files: set):
        """Delete documents from the index if their corresponding files have been deleted from the filesystem."""
        ref_doc_info = self.index.ref_doc_info
        deleted_docs = []

        for doc_id, info in ref_doc_info.items():
            file_path = info.metadata.get("file_path")
            if file_path and os.path.abspath(file_path) not in current_files:
                self.logger.info(f"Deleted file: {file_path}")
                deleted_docs.append(doc_id)

        if deleted_docs:
            self.logger.info(f"Deleting {len(deleted_docs)} chunks from the index.")
            for doc_id in deleted_docs:
                self.index.delete_ref_doc(doc_id, delete_from_docstore=True)
            self.logger.info("Deletion process completed.")
        else:
            self.logger.info("No deleted files found.")

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        # Count unique file paths in the index
        unique_files = set()
        for doc_id, info in self.index.ref_doc_info.items():
            file_path = info.metadata.get("file_path")
            if file_path:
                unique_files.add(file_path)

        return {
            "num_documents": len(unique_files),  # Count of unique documents
            "num_nodes": len(self.index.ref_doc_info),  # Total number of nodes
            "index_persist_dir": self.index_persist_dir,
            "chroma_db_dir": self.chroma_db_dir,
            "chroma_collection_name": self.chroma_collection_name,
        }
