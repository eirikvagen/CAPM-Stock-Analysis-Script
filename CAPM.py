"""
Class that automatically is able to run capm 3 factor model automatically
"""
import pandas as pd
import numpy as np
from numpy import nan
import yfinance as yf
import matplotlib.pyplot as plt
import pandas_datareader.data as reader
import datetime as dta
from sklearn import linear_model
import statsmodels.api as sm
import csv


class PortfolioAnalysis:
    def __init__(self):
        self.class_df = pd.DataFrame()

    def setup_data(
        self, tickers: list[str], start: str = "2015-01-01", end: str = "2022-01-01"
    ):
        # Tidsperiode
        self.class_df = yf.download(tickers, start, end, interval="1d", threads=True)[
            "Adj Close"
        ]

    def fix_data_or_something(self, SMALong: int = 200, SMAshort: int = 50):
        df_bool_short_above_long = (
            self.class_df.rolling(window=SMAshort).mean()
            > self.class_df.rolling(window=SMALong).mean()
        )

        # TODO remove df temporary
        df_temporary = df_bool_short_above_long.mask(
            df_bool_short_above_long is False, np.nan
        )
        df_temporary["composition"] = df_temporary.apply(
            lambda row: ",".join(row.index[row is True]), axis=1
        )

        # logchanges from day to day
        logreturns = np.log(1 + self.class_df.pct_change())
        # composition of stocks added to logreturns
        logreturns["innhold"] = df_temporary["composition"]
        # column for sum of the daily log returns
        logreturns["summerteLogreturns"] = 0
        # logreturns without composition
        logreturnsUtenInnhold = logreturns.drop("innhold", axis=1)

        # Used to check the previous composition
        prevInnhold = ""
        # counts the number of changes in the composition
        number_of_composition_changes = 0

        # iterate the rows of the logreturns
        for index, row in logreturns.iterrows():
            # list of the composition in the composition column
            innhold = logreturns.loc[index, "innhold"]
            innhold = innhold.split(",")
            # placeholdervalue for the sum of returns of the row
            midlertidiglogreturn = 0

            # Iterate the cells of the row
            for col_name, cell_value in row.iteritems():
                # If the column is in the composition and innhold is larger than 1 it will sum the return.
                # TODO this will not add the return when there is only one stock in composition
                if (col_name in innhold) and (len(innhold) > 1):
                    # TODO should not be - 1??, because len([a, b]) = 2, not 3.
                    midlertidiglogreturn += cell_value / (len(innhold) - 1)

                # buy and sell adds a composition change if the stock is taken in or out of out of the composition
                # buy
                if (col_name in innhold) and (col_name not in prevInnhold):
                    number_of_composition_changes += 1

                # sell
                if (col_name not in innhold) and (col_name in prevInnhold):
                    number_of_composition_changes += 1

            # lastly for the row iterater, it will put the sum of the returns into the column for summed logreturns
            logreturnsUtenInnhold.loc[
                index, "summerteLogreturns"
            ] += midlertidiglogreturn

            # sets the composition variable to hold the previous composition for the next iteration to check for compostion changes
            prevInnhold = innhold

        # Cumulates the returns of all the rows and change the values back to arithmetic numbers
        logreturnsUtenInnhold["summerteLogreturns"] = np.exp(
            np.log(1 + logreturnsUtenInnhold["summerteLogreturns"]).cumsum()
        )
        logreturnsUtenInnhold.index = logreturnsUtenInnhold.index.strftime("%Y-%m-%d")

    def capm_reg_data(self):
        # gets capm regnumbers for smb hml and momentum
        df = pd.read_csv("../data_til_analyse/daglige_tall_CAPM.csv", index_col="date")
        df.index = pd.to_datetime(df.index, format="%Y%m%d", errors="coerce")
        portfolioDF = pd.DataFrame(columns=["portfolio"], index=["date"])

        # logchanges from day to day
        logreturns = np.log(1 + self.class_df.pct_change())
        # composition of stocks added to logreturns
        logreturns["innhold"] = df_temporary["composition"]
        # column for sum of the daily log returns
        logreturns["summerteLogreturns"] = 0
        # logreturns without composition
        logreturnsUtenInnhold = logreturns.drop("innhold", axis=1)

        rows_to_concat = []
        for index, row in logreturnsUtenInnhold.iterrows():
            if index in df.index:
                indeks = index
                new_row = {
                    indeks: logreturnsUtenInnhold.loc[index, "summerteLogreturns"]
                }
                rows_to_concat.append(
                    pd.DataFrame.from_dict(
                        new_row, orient="index", columns=["portfolio"]
                    )
                )

        if rows_to_concat:
            portfolioDF = pd.concat([portfolioDF] + rows_to_concat)

        portfolioDF.index.name = "date"
        portfolioDF = portfolioDF.drop("date")
        portfolioDF = portfolioDF.pct_change()
        portfolioDF = portfolioDF[SMALong:]
        df = pd.read_csv("./data_til_analyse/daglige_tall_CAPM.csv", index_col="date")

        df = df[["SMB", "HML"]]

        # risk free rate for capm
        rf = pd.read_csv("./data_til_analyse/rf_daglig.csv", index_col="date")

        new_df = df.copy()
        new_df["rf"] = 0

        # Convert the index to datetime objects
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        rf.index = pd.to_datetime(rf.index, format="%Y-%m-%d")

        # Merge the two DataFrames based on the index
        df_mergedPricingFactors = pd.merge(
            df, rf, left_index=True, right_index=True, how="inner"
        )
        merged_df = pd.merge(
            left=df_mergedPricingFactors,
            right=portfolioDF,
            left_index=True,
            right_index=True,
            how="inner",
        )
        df_mergedPricingFactors.index = pd.to_datetime(
            df_mergedPricingFactors.index, format="%Y%m%d"
        )

        for index, row in portfolioDF.iterrows():
            if index in df_mergedPricingFactors.index:
                new_row = {
                    "SMB": df_mergedPricingFactors.loc[indeks, "SMB"],
                    "HML": df_mergedPricingFactors.loc[indeks, "HML"],
                    "rf(1d)": df_mergedPricingFactors.loc[indeks, "rf(1d)"],
                    "portfolio": portfolioDF.loc[indeks, "portfolio"],
                }
                new_df = pd.DataFrame(
                    new_row,
                    index=[index],
                    columns=["SMB", "HML", "rf(1d)", "portfolio"],
                )
                merged_df = pd.concat([merged_df, new_df])

        mr = pd.read_csv("./dataset/market_portfolios_daily.csv", index_col="date")
        mr = mr.drop("date")
        mr = mr["OSEAX"]
        mr.index = pd.to_datetime(mr.index, format="%Y%m%d")
        merged_df = pd.merge(
            left=merged_df, right=mr, left_index=True, right_index=True, how="inner"
        )

        try:
            portfolioDF = portfolioDF.drop("2017-04-13")
        except KeyError:
            print("")
        try:
            portfolioDF = portfolioDF.drop("2004-02-13")
        except KeyError:
            print("")
        try:
            portfolioDF = portfolioDF.drop("2004-02-18")
        except KeyError:
            print("")
        try:
            portfolioDF = portfolioDF.drop("2004-10-01")
        except KeyError:
            print("")
        try:
            portfolioDF = portfolioDF.drop("2005-09-05")
        except KeyError:
            print("")
        try:
            portfolioDF = portfolioDF.drop("2009-04-28")
        except KeyError:
            print("")
        try:
            portfolioDF = portfolioDF.drop("2009-11-25")
        except KeyError:
            print("")

        for index, row in portfolioDF.iterrows():
            if index in mr.index:
                new_row = {
                    "OSEAX": mr.loc[index],
                    "SMB": df_mergedPricingFactors.loc[index, "SMB"],
                    "HML": df_mergedPricingFactors.loc[index, "HML"],
                    "rf(1d)": df_mergedPricingFactors.loc[index, "rf(1d)"],
                    "portfolio": portfolioDF.loc[index, "portfolio"],
                }
                new_df = pd.DataFrame(
                    new_row,
                    index=[index],
                    columns=["SMB", "HML", "rf(1d)", "portfolio", "OSEAX"],
                )
                merged_df = pd.concat([merged_df, new_df])

        # change name of rf to be compatible further on
        merged_df = merged_df.rename(columns={"rf(1d)": "rf"})

    def capm_regression(self, df: pd.DataFrame):
        df["OSEAX"] = df["OSEAX"].astype(float)
        df["portfolio-rf"] = df.portfolio - df.rf
        df["MRKT-rf"] = df.OSEAX - df.rf

        Y = df["portfolio-rf"]
        X = df[["MRKT-rf", "SMB", "HML"]]

        X_sm = sm.add_constant(X)

        model = sm.OLS(Y, X_sm)
        results = model.fit()
        results.summary()

    def save_results(self):
        # Your code for saving results to a CSV file
        pass


if __name__ == "__main__":
    tickersobx = pd.read_html("https://no.wikipedia.org/wiki/OSEBX-indeksen")[0]
    tickersobx = tickersobx["Ticker"].to_list()
    tickersobx = [i.replace("OSE: ", "") for i in tickersobx]
    tickersobx = [i + (".OL") for i in tickersobx]
    tickersobx = [i.replace("TIETOO.OL", "TIETO.OL") for i in tickersobx]
    tickersobx = [i.replace("SCHB.OL", "SCHBA.OL") for i in tickersobx]
    tickers_to_remove = [
        "FJORD.OL",
        "SRBANK.OL",
        "NOFI.OL",
        "AIRX.OL",
        "AKH.OL",
        "AGLX.OL",
    ]
    tickersobx = [ticker for ticker in tickersobx if ticker not in tickers_to_remove]

    portfolio_analyzer = PortfolioAnalysis()
    portfolio_analyzer.setup_data(tickersobx)
    portfolio_analyzer.run_analysis()
    portfolio_analyzer.save_results()
