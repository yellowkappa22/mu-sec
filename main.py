from typing import List
from ib_async import *

DUR_GRAN_CONSTRAINTS = {
    1800: 1, # 30m | 1s
    3600: 5, # 1h | 5s
    14400: 10, # 4h | 10s
    28800: 30, # 8h | 30s
    86400: 60, # 1d | 1m
    172800: 120, # 2d | 2m
    604800: 180, # 1w | 3m
    2592000
}

class IBClient:
    def __init__(self, host_ip: str, port: int):
        ib = IB()
        ib.connect(host_ip, port, clientId=1)

    def req_dataset(self, assets: List[str], )

contract = Forex('EURUSD')
bars = ib.reqHistoricalData(
    contract, endDateTime='', durationStr='30 D',
    barSizeSetting='1 hour', whatToShow='MIDPOINT', useRTH=True)

df = util.df(bars)
print(df)
if __name__ == '__main__':
    ib_client = IBClient(
        host_ip='172.18.144.1',
        port=7496
    )