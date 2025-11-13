import click

@click.group()
def cli():
    pass

@cli.command()
def ingest():
    """
    Run the data ingestion pipeline.
    """
    click.echo("Running data ingestion pipeline...")
    # TODO: Add call to scripts/ingest.py
    click.echo("Data ingestion complete.")

@cli.command()
def api():
    """
    Run the retrieval API.
    """
    click.echo("Running retrieval API...")
    # TODO: Add call to run the FastAPI app
    click.echo("Retrieval API stopped.")

if __name__ == '__main__':
    cli()