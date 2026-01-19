from typing import List, Literal
from ib_async import util, IB, Stock

class IBClient:
    def __init__(self, host: str, port: int, client_id=1):
        self.ib = IB()
        self.ib.connect(host, port, clientId=client_id)

    def get_stock_data(self, 
                        tickers: List[str], # ex. ['AAPL']
                        period_days: int,
                        bar_size: Literal['1 min', '1 hour', '1 day'],
                        end_time="" # datetime
        ) -> List[pd.DataFrame]:
        #
        stock = Stock(ticker)
        durationStr = f"{period_days} D"
        #
        stocks = {}
        for ticker in tickers:
            stock_data = self.ib.reqHistoricalData(stock, endDateTime=end_time, durationStr=durationStr, barSizeSetting=bar_size, whatToShow='TRADES', useRTH=True)
            stock_df = util.df(stock_data)
            stocks[ticker] = (stock_df, stock_df.shape[0])
        #
        return stocks
    
    def req_exchange(self,
            ticker: str
        ) -> str:
        #
        stock = Stock(ticker)
        contract_details = self.ib.reqContractDetails(stock)
        exchange = contract_details[0].contract.primaryExchange
        #
        return exchange
        