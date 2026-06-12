"""VRB 0DTE option backtesting, forecasting, and prediction-modeling toolkit.

Data sources (see vrb/config.py for paths):
  - IB TWS 1-sec OHLCV bars: ES, NQ (futures), SPX, NDX (cash) + 1-min VIX
  - ThetaData 5-sec NBBO option quotes: SPXW (and NDX once downloaded)

All timestamps everywhere are naive US/Central strings or datetime64, per the
conventions documented in the downloader repo's understanding_this_data.md:
same timestamp value = same instant across every file.
"""

__version__ = "0.1.0"
