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
# Abstract Base Handler
# -------------------------------------------------------------------
class DataFrameHandler(ABC):
    """
    Interface for all data transformation components.
    """
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and return a pandas DataFrame"""
        pass


# -------------------------------------------------------------------
# Concrete Handler: Missing Values & Duplicates
# -------------------------------------------------------------------
class MissingAndDuplicateHandler(DataFrameHandler):
    """
    Handles basic data cleaning by removing null values and duplicate records.
    """
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops missing values and duplicate rows from the DataFrame.
        """
        try:
            rows_before = len(df)
            logger.info(f"🧹 Starting data cleaning process | Initial rows: {rows_before}")

            # Step 1: Drop missing values
            df = df.dropna()
            
            # Step 2: Drop duplicate rows
            df = df.drop_duplicates()

            rows_after = len(df)
            logger.info(f"✨ Data cleaning completed | Final rows: {rows_after}")
            logger.info(f"✅ Summary: Removed {rows_before - rows_after} rows in total.")

            return df

        except Exception as e:
            logger.error(f"❌ Error during missing/duplicate handling: {e}")
            raise e
