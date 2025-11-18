"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=120, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma
        self.ma_window = 200  # for moving average filter

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        # 取 SPY 價格與 200 日均線
        spy_price = self.price[self.exclude]
        spy_ma = spy_price.rolling(self.ma_window).mean()

        # 從足夠資料開始（同時滿足 lookback 與 ma_window）
        start_idx = max(self.lookback, self.ma_window)

        for i in range(start_idx, len(self.price)):
            idx = self.price.index[i]

            # 每天先把權重清成 0，避免 ffill 造成 leverage
            self.portfolio_weights.loc[idx] = 0.0

            # 如果 SPY 處於風險 OFF（跌破長期均線），就完全空倉
            if pd.isna(spy_ma.iloc[i]) or spy_price.iloc[i] <= spy_ma.iloc[i]:
                # 全部維持 0 權重（持現金）
                continue

            # 風險 ON：計算過去 lookback 天的動能
            window_prices = self.price[assets].iloc[i - self.lookback : i]
            # 避免除 0 之類問題，確保沒有空視窗
            if window_prices.isnull().values.any():
                window_prices = window_prices.fillna(method="ffill").fillna(method="bfill")

            momentum = window_prices.iloc[-1] / window_prices.iloc[0] - 1

            # 依動能排序，取前 3 名
            top_assets = momentum.sort_values(ascending=False).head(3).index

            # 等權重分配給 top-3
            w = 1.0 / len(top_assets)
            self.portfolio_weights.loc[idx, top_assets] = w
            # 確保 SPY 權重為 0
            self.portfolio_weights.loc[idx, self.exclude] = 0.0
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
