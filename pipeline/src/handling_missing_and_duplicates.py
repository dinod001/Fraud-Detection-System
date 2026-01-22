import logging
import pandas as pd
from abc import ABC, abstractmethod

# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Abstract base handler
# -------------------------------------------------------------------
class DataFrameHandler(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and return a pandas DataFrame"""
        pass


# -------------------------------------------------------------------
# Concrete handler: missing values & duplicates
# -------------------------------------------------------------------
class MissingAndDuplicateHandler(DataFrameHandler):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            rows_before = len(df)
            logger.info(f"Starting data cleaning. Rows before: {rows_before}")

            # Drop missing values
            df = df.dropna()

            # Drop duplicate rows
            df = df.drop_duplicates()

            rows_after = len(df)
            logger.info(f"Data cleaning completed. Rows after: {rows_after}")
            logger.info(f"Rows removed: {rows_before - rows_after}")

            return df

        except Exception as e:
            logger.exception("Error occurred while handling missing values and duplicates")
            raise e
