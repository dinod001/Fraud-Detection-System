import logging
from abc import ABC, abstractmethod
import pandas as pd

# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Abstract Base Class
# -------------------------------------------------------------------
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path_or_link: str) -> pd.DataFrame:
        """Ingest data from a given file path or URL."""
        pass


# -------------------------------------------------------------------
# CSV Ingestor
# -------------------------------------------------------------------
class DataIngestorCSV(DataIngestor):
    def ingest(self, file_path_or_link: str) -> pd.DataFrame:
        logger.info(f"Starting CSV ingestion from: {file_path_or_link}")
        try:
            df = pd.read_csv(file_path_or_link)

            rows, cols = df.shape
            logger.info(
                f"CSV ingestion successful | Rows: {rows}, Columns: {cols}"
            )

            return df
        except Exception:
            logger.error("CSV ingestion failed", exc_info=True)
            raise


# -------------------------------------------------------------------
# Excel Ingestor
# -------------------------------------------------------------------
class DataIngestorExcel(DataIngestor):
    def ingest(self, file_path_or_link: str) -> pd.DataFrame:
        logger.info(f"Starting Excel ingestion from: {file_path_or_link}")
        try:
            df = pd.read_excel(file_path_or_link)

            rows, cols = df.shape
            logger.info(
                f"Excel ingestion successful | Rows: {rows}, Columns: {cols}"
            )

            return df
        except Exception:
            logger.error("Excel ingestion failed", exc_info=True)
            raise


#calling
excel = DataIngestorCSV()
excel.ingest("data/raw/Fraud_Data.csv")