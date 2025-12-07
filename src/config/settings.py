import json
from pathlib import Path

from pydantic_settings import BaseSettings
from tinydb import Query
from tinydb import TinyDB

DB_PATH = Path("data/projects.json")


class ProjectManager:
    """Manages project state and tracking using TinyDB"""

    def __init__(self):
        # Ensure data directory exists
        Path("data").mkdir(exist_ok=True)
        self.db = TinyDB(DB_PATH)
        self.query = Query()

    def list_projects(self) -> list[dict]:
        """List all tracked projects"""
        return self.db.all()

    def get_project_by_path(self, project_path: str) -> dict | None:
        """Get a project by its path"""
        return self.db.get(self.query.project_path == project_path)

    def get_project_by_id(self, project_id: int) -> dict | None:
        """Get a project by its ID"""
        return self.db.get(self.query.id == project_id)

    def add_project(self, project_path: str, data_dir: str) -> int:
        """Add a new project to tracking"""
        project = {
            "project_path": project_path,
            "data_directory": data_dir,
            "indexed_at": None,
            "status": "created",
            "pipeline_state": {"parsed": False, "preprocessed": False, "embedded": False, "indexed": False},
        }
        return self.db.insert(project)

    def update_project_status(self, project_id: int, status: str):
        """Update project status"""
        self.db.update({"status": status}, self.query.id == project_id)

    def update_pipeline_state(self, project_id: int, step: str, *, completed: bool):
        """Update pipeline state for a project (completed must be passed as a keyword argument)"""
        project = self.db.get(self.query.id == project_id)
        if project:
            pipeline_state = project.get("pipeline_state", {})
            pipeline_state[step] = completed
            self.db.update({"pipeline_state": pipeline_state}, self.query.id == project_id)


class ProjectError(ValueError):
    """Custom exception for errors in the application configuration."""

    def __init__(self, message="Project not initialized. Call initialize_project() first."):
        self.message = message
        super().__init__(self.message)


class Settings(BaseSettings):
    # Embedding Configuration
    embedding_model_url: str
    embedding_model_name: str
    embedding_dim: int

    # LLM Generation Configuration
    generation_model_url: str
    generation_model_name: str
    generation_model_provider: str

    # Qdrant Configuration
    qdrant_host: str
    qdrant_port: int
    qdrant_api_key: str

    # Neo4j Configuration
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str

    # Project path provided during execution
    project_path: Path | None = None

    # Runtime-computed project-specific values
    project_data_dir: Path | None = None
    project_id: str | None = None
    project_name: str | None = None
    project_db_id: int | None = None

    # Project manager instance
    project_manager: ProjectManager = ProjectManager()

    class Config:
        env_file = ".env"

    def initialize_project(self, project_path: str | Path):
        """
        Initialize project-specific values based on the provided path
        If the project already exists in the database, load its information
        Otherwise, create a new project entry and associated directories
        """
        path = Path(project_path)

        # Use relative path if it's not already absolute
        if not path.is_absolute():
            path = Path.cwd() / path

        # Check if project already exists in the database
        existing_project = self.project_manager.get_project_by_path(str(path))

        if existing_project:
            # Load existing project information
            self.project_path = path
            self.project_data_dir = Path(existing_project["data_directory"])
            self.project_id = existing_project.get("project_id")  # This might not exist in current DB
            self.project_name = path.name
            self.project_db_id = existing_project.doc_id
        else:
            # Create new project
            self.project_path = path

            # Use project name only for ID (no more SHA ID)
            self.project_id = path.name
            self.project_name = path.name

            # Create a project-specific data directory using relative path
            data_dir = Path("data")
            project_data_dir = data_dir / f"project_{self.project_name}"
            project_data_dir.mkdir(parents=True, exist_ok=True)
            self.project_data_dir = project_data_dir

            # Add project to the database
            self.project_db_id = self.project_manager.add_project(str(path), str(project_data_dir))

    def get_project_config_path(self) -> Path:
        """
        Get the path to the project-specific configuration file
        """
        if self.project_data_dir is None:
            raise ProjectError
        return self.project_data_dir / "project_config.json"

    def get_chunks_path(self) -> Path:
        """
        Get the path to the chunks.json file for the current project
        """
        if self.project_data_dir is None:
            raise ProjectError
        return self.project_data_dir / "chunks.json"

    def get_symbol_index_path(self) -> Path:
        """
        Get the path to the chunks.json file for the current project
        """
        if self.project_data_dir is None:
            raise ProjectError
        return self.project_data_dir / "symbol_index.json"

    def get_enriched_chunks_path(self) -> Path:
        """Get path to summarized chunks"""
        if self.project_data_dir is None:
            raise ProjectError
        return self.project_data_dir / "enriched_chunks.json"

    def get_embedded_chunks_path(self) -> Path:
        """
        Get the path to the embedded_chunks.json file for the current project
        """
        if self.project_data_dir is None:
            raise ProjectError
        return self.project_data_dir / "embedded_chunks.json"

    def get_preprocessed_chunks_path(self) -> Path:
        """
        Get the path to the preprocessed_chunks.json file for the current project
        """
        if self.project_data_dir is None:
            raise ProjectError
        return self.project_data_dir / "preprocessed_chunks.json"

    def get_project_db_path(self) -> Path:
        """
        Get the path to the project-specific tinyDB file for additional project data
        """
        if self.project_data_dir is None:
            raise ProjectError
        return self.project_data_dir / "enhanced_chunks.db"

    def save_project_config(self):
        """
        Save project-specific configuration to a file
        """
        if self.project_data_dir is None:
            raise ProjectError

        config_data = {
            "project_path": str(self.project_path),
            "project_id": self.project_id,
            "project_name": self.project_name,
            "data_directory": str(self.project_data_dir),
            "indexed_at": self.project_manager.get_project_by_id(self.project_db_id).get("indexed_at"),
            "pipeline_state": self.project_manager.get_project_by_id(self.project_db_id).get("pipeline_state"),
        }

        config_path = self.get_project_config_path()
        with Path(config_path).open("w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, default=str)  # default=str handles datetime objects


settings = Settings()
