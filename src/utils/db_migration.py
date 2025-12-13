# ruff :noqa
"""
Database migration script to transfer summary field data from enhanced_chunks.db
to ./data/project_name/enhanced_chunks.db based on semantic hash matching.
"""

import json
import sqlite3
from pathlib import Path


def migrate_summary_field(source_db_path: str, target_db_path: str, project_name: str = None):
    """
    Migrate summary field from source DB to target DB based on semantic hash matching.

    Args:
        source_db_path: Path to the source database (enhanced_chunks.db)
        target_db_path: Path to the target database (./data/project_name/enhanced_chunks.db)
        project_name: Optional project name to use in target path (if not provided in target_db_path)
    """
    source_conn = None
    target_conn = None

    try:
        # Connect to both databases
        source_conn = sqlite3.connect(source_db_path)
        target_conn = sqlite3.connect(target_db_path)

        # Get tables from both databases
        source_cursor = source_conn.cursor()
        target_cursor = target_conn.cursor()

        # Find tables that contain 'analysis' column with 'semantic_hash'
        source_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        source_tables = [row[0] for row in source_cursor.fetchall()]

        target_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        target_tables = [row[0] for row in target_cursor.fetchall()]

        # Find matching tables between source and target
        matching_tables = set(source_tables) & set(target_tables)

        if not matching_tables:
            print("No matching tables found between source and target databases.")
            return

        for table_name in matching_tables:
            print(f"Processing table: {table_name}")

            # Check if 'summary' column exists in source table
            source_cursor.execute(f"PRAGMA table_info({table_name})")
            source_columns = [column[1] for column in source_cursor.fetchall()]

            # Check if 'summary' column exists in target table
            target_cursor.execute(f"PRAGMA table_info({table_name})")
            target_columns = [column[1] for column in target_cursor.fetchall()]

            # If 'summary' column doesn't exist in target, add it
            if "summary" not in target_columns:
                target_cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN summary TEXT")
                target_conn.commit()
                print(f"Added 'summary' column to {table_name} in target database")

            # Get all records from source that have both analysis field with semantic_hash and summary field
            source_cursor.execute(f"""
                SELECT rowid, analysis, summary
                FROM {table_name}
                WHERE analysis IS NOT NULL
                AND analysis != ''
                AND summary IS NOT NULL
                AND summary != ''
            """)

            source_records = source_cursor.fetchall()

            # Create a mapping of semantic_hash to summary from source
            semantic_hash_to_summary = {}
            for source_rowid, analysis_json, summary in source_records:
                try:
                    analysis = json.loads(analysis_json)
                    if "semantic_hash" in analysis:
                        semantic_hash = analysis["semantic_hash"]
                        if summary:  # Only store non-empty summaries
                            semantic_hash_to_summary[semantic_hash] = summary
                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue

            # Get records from target that have analysis field with semantic_hash
            target_cursor.execute(f"""
                SELECT rowid, analysis
                FROM {table_name}
                WHERE analysis IS NOT NULL
                AND analysis != ''
            """)

            target_records = target_cursor.fetchall()

            # Update target records based on semantic hash match
            updated_count = 0
            for target_rowid, target_analysis_json in target_records:
                try:
                    target_analysis = json.loads(target_analysis_json)
                    if "semantic_hash" in target_analysis:
                        target_hash = target_analysis["semantic_hash"]

                        # If we have a summary for this semantic hash
                        if target_hash in semantic_hash_to_summary:
                            # Update the summary column in target DB
                            summary = semantic_hash_to_summary[target_hash]
                            target_cursor.execute(
                                f"UPDATE {table_name} SET summary = ? WHERE rowid = ?",
                                (summary, target_rowid),
                            )
                            updated_count += 1

                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue

            print(f"Updated {updated_count} records in table '{table_name}'")

        # Commit changes
        target_conn.commit()
        print("Migration completed successfully!")

    except Exception as e:
        print(f"Error during migration: {e}")
        if target_conn:
            target_conn.rollback()
        raise
    finally:
        if source_conn:
            source_conn.close()
        if target_conn:
            target_conn.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Migrate summary field between databases")
    parser.add_argument("--source", required=True, help="Path to source database (enhanced_chunks.db)")
    parser.add_argument(
        "--target", required=True, help="Path to target database (./data/project_name/enhanced_chunks.db)"
    )
    parser.add_argument("--project-name", help="Project name to use in target path (optional)")

    args = parser.parse_args()

    # Validate source exists
    if not Path(args.source).exists():
        print(f"Source database does not exist: {args.source}")
        return 1

    # Create target directory if it doesn't exist
    target_path = Path(args.target)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    migrate_summary_field(args.source, args.target, args.project_name)
    return 0


if __name__ == "__main__":
    exit(main())
