"""
Neo4j configuration with connectivity check
"""

from contextlib import suppress

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
        driver = None
        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            with driver.session() as session:
                # Run a simple query to test connectivity
                result = session.run("RETURN 1 AS test")
                record = result.single()
                return bool(record and record["test"] == 1)
        except Exception:
            return False
        finally:
            if driver:
                with suppress(BaseException):
                    driver.close()
