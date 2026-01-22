import logging
import pandas as pd
from abc import ABC, abstractmethod

# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Abstract Base Class
# -------------------------------------------------------------------
class DataIngestor(ABC):
    """
    Interface for data ingestion strategies.
    Any new data source (SQL, API, etc.) should implement this interface.
    """
    @abstractmethod
    def ingest(self, file_path_or_link: str) -> pd.DataFrame:
        """Loads data into a pandas DataFrame."""
        pass


# -------------------------------------------------------------------
# CSV Ingestion Implementation
# -------------------------------------------------------------------
class DataIngestorCSV(DataIngestor):
    """
    Strategy for ingesting data from CSV files.
    """
    def ingest(self, file_path_or_link: str) -> pd.DataFrame:
        """
        Reads a CSV file from a local path or URL.
        """
        logger.info(f"📂 Starting CSV ingestion process from: {file_path_or_link}")
        try:
            df = pd.read_csv(file_path_or_link)

            rows, cols = df.shape
            logger.info(f"✅ CSV Ingestion successful | Shape: ({rows} rows, {cols} columns)")

            return df
        except Exception as e:
            logger.error(f"❌ Critical failure during CSV ingestion: {e}")
            raise e


# -------------------------------------------------------------------
# Excel Ingestion Implementation
# -------------------------------------------------------------------
class DataIngestorExcel(DataIngestor):
    """
    Strategy for ingesting data from Excel files (.xlsx, .xls).
    """
    def ingest(self, file_path_or_link: str) -> pd.DataFrame:
        """
        Reads an Excel spreadsheet from a local path or URL.
        """
        logger.info(f"📊 Starting Excel ingestion process from: {file_path_or_link}")
        try:
            df = pd.read_excel(file_path_or_link)

            rows, cols = df.shape
            logger.info(f"✅ Excel Ingestion successful | Shape: ({rows} rows, {cols} columns)")

            return df
        except Exception as e:
            logger.error(f"❌ Critical failure during Excel ingestion: {e}")
            raise e