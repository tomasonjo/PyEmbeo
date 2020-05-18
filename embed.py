from embeoj.export import export
from embeoj.preprocess import preprocess_exported_data
from embeoj.train import convert_tsv_to_pbg, train_embeddings
from embeoj.utils import update_config, test_db_connection, logging
import click
import sys

def embed():
    """Command line interface for training and generating graph embeddings
    """
    try:
        # test run to check for db connection
        if not test_db_connection():
            logging.info("could not connect to Neo4j")
            sys.exit()
            return

        export()  # export graph data to tsv json file
        preprocess_exported_data()  # convert to tsv file for biggraph to read
        convert_tsv_to_pbg()  # process data files for training
        train_embeddings()  # train
        logging.info("Done....")

    except Exception as e:
        logging.info(f"Error: {e}")
        sys.exit(e)


if __name__ == "__main__":
    embed()
