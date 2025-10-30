#!/usr/bin/env python3
"""
explore_parquet.py

Quick exploration tool for a LeRobot-format dataset stored in Parquet format.

Usage:
    python explore_parquet.py path/to/dataset.parquet
"""

import sys
import pandas as pd


def main(parquet_path: str):
    # ---------------------------------------------------------------------
    # 1. Load parquet file into a DataFrame
    # ---------------------------------------------------------------------
    print(f"\n📂 Loading Parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"✅ Loaded dataset with {len(df):,} rows and {len(df.columns)} columns\n")

    # ---------------------------------------------------------------------
    # 2. Basic info and structure
    # ---------------------------------------------------------------------
    print("=== Basic Info ===")
    print(df.info())  # shows column types and memory usage
    print("\n=== Column names ===")
    print(df.columns.tolist())

    # ---------------------------------------------------------------------
    # 3. Peek at the first few rows
    # ---------------------------------------------------------------------
    print("\n=== Head (first 5 rows) ===")
    print(df.head())

    # ---------------------------------------------------------------------
    # 4. Basic descriptive statistics
    # ---------------------------------------------------------------------
    print("\n=== Descriptive statistics (numeric columns) ===")
    print(df.describe())

    # ---------------------------------------------------------------------
    # 5. Sample exploration helpers
    # ---------------------------------------------------------------------
    print("\n=== Example helper functions ===")

    def show_random_sample(n: int = 3):
        """Print a random sample of rows."""
        print(f"\n🎲 Random sample ({n} rows):")
        print(df.sample(n))

    def show_unique_values(col_name: str):
        """Show unique values in a given column."""
        if col_name not in df.columns:
            print(f"❌ Column '{col_name}' not found.")
            return
        uniques = df[col_name].unique()
        print(f"\n🧩 Unique values in '{col_name}' ({len(uniques)}):")
        print(uniques[:20])  # show up to 20

    def summarize_column(col_name: str):
        """Print basic statistics for a numeric column."""
        if col_name not in df.columns:
            print(f"❌ Column '{col_name}' not found.")
            return
        print(f"\n📈 Summary of '{col_name}':")
        print(df[col_name].describe())

    # Example usage of the helper functions
    show_random_sample(2)
    # if len(df.columns) > 0:
    #     show_unique_values(df.columns[0])
    #     summarize_column(df.columns[0])

    print("\n💡 You can now import this script in a Jupyter notebook or REPL to play with the df object directly.")
    print("   For example:")
    print("      >>> from explore_parquet import main")
    print("      >>> df = pd.read_parquet('path/to/dataset.parquet')")
    print("      >>> df.head()")


if __name__ == "__main__":


    parquet_path = "/home/nimrod/dev/so101_rosbag2lerobot_dataset/local_dataset_ros2/data/data/chunk-000/file-000.parquet"
    main(parquet_path)
