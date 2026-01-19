from pyspark.sql import SparkSession
from datetime import datetime
from delta import configure_spark_with_delta_pip
from clients.ibkr_client import IBClient
from dotenv import load_dotenv
import os
from typing import Literal

load_dotenv()

class DataManager:
    def __init__(self):
        builder = SparkSession.builder.appName("trading_delta_lake") \
                    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        self.spark = configure_spark_with_delta_pip(builder).getOrCreate()
        self.delta_path = os.getenv("DELTA_PATH")
        self.ib_client = IBClient(os.getenv("IBKR_HOST"), os.getenv("IBKR_PORT"))
        
    def cache_data(self, path: str, df: pyspark.sql.DataFrame):
        path = os.path.join(self.delta_path, cache_name)
        df.write.mode("overwrite").format("delta").save(path)
        
    def get_stock_data(self,
                       tickers: List[str],
                       start_time: datetime,
                       end_time: datetime,
                       bar_size: Literal['1 min', '1 hour', '1 day']
                    ):
        
        
        path = os.path.join(self.delta_path, granularity)
        if not os.path.isdir(path):
            print(f"data repository {path} does not exist under delta")
            return None
        try:
            df = self.spark.read.format("delta").load(path)
            df = df.filter(
                (col("ticker").isin(tickers)) &
                (col("timestamp") >= start_time) &
                (col("timestamp") <= end_time)
            )