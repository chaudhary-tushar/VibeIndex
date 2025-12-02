"""
Neo4j configuration with connectivity check
"""

from neo4j import GraphDatabase
from pydantic import ConfigDict
from pydantic import Field
from pydantic_settings import BaseSettings


class Neo4jConfig(BaseSettings):
    """
    Configuration for Neo4j graph database
    """

    uri: str = Field(..., description="Neo4j connection URI (e.g., bolt://localhost:7687)")
    username: str = Field(..., description="Neo4j username")
    password: str = Field(..., description="Neo4j password")

    model_config = ConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    def ping(self) -> bool:
        """
        Check if the Neo4j service is reachable by attempting a simple connection
        """
        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            with driver.session() as session:
                # Run a simple query to test connectivity
                result = session.run("RETURN 1 AS test")
                record = result.single()
                if record and record["test"] == 1:
                    return True
                return False
        except Exception:
            return False
        finally:
            try:
                driver.close()
            except:
                pass  # Driver might already be closed
