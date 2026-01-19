import numpy as np
from time import time
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple

# test data setup
ASSET_MAP = {
    "AAPL": 0,
    "NVDA": 1
}
FEATURE_MAP = {
    "timestamp": 0,
    "open_": 1,
    "high_": 2,
    "low_": 3,
    "close_": 4
}
DELTA = 24 # granuality in hours
TEST_DATA = np.array(
    [
        [
            [
                datetime(2026, 1, 8), 8, 10, 5, 7
            ],
            [
                datetime(2026, 1, 8), 3, 5, 3, 4
            ]
        ],
        [
            [
                datetime(2026, 1, 9), 7, 7, 5, 4
            ],
            [
                datetime(2026, 1, 9), 4, 6, 4, 5
            ]
        ],
        [
            [
                datetime(2026, 1, 10), 4, 5, 2, 3
            ],
            [
                datetime(2026, 1, 10), 5, 9, 4, 7
            ]
        ]
    ]
)

# event classes setup
@dataclass
class DataEvent:
    asset: str
    timestamp: datetime
    open_: float
    high_: float
    low_: float
    close_: float
    
# data handler
class HistoricDataHandler:
    def __init__(self, data, asset_map, feature_map):
        self.data = data
        self.data_idx = 0
        self.asset_map = asset_map
        self.feature_map = feature_map
    
    def get(self, assets: List[str], features: List[str], n):
        asset_mask = [asset_map[asset] for asset in assets]
        feature_mask = [feature_map[feature] for feature in features]
        latest_data = data[(self.data_idx - n):self.data_idx, asset_mask, feature_mask]
        if len(latest_data) == n:
            return latest_data
        else:
            return None
        
    def update(self, events: deque):
        if self.data_idx >= self.data.shape[0]:
            return False
        for asset in self.asset_map.keys():
            new_event = DataEvent(
                asset=asset,
                timestamp=self.data[self.data_idx, self.asset_map[asset], self.feature_map["timestamp"]],
                open_=self.data[self.data_idx, self.asset_map[asset], self.feature_map["open_"]],
                high_=self.data[self.data_idx, self.asset_map[asset], self.feature_map["high_"]],
                low_=self.data[self.data_idx, self.asset_map[asset], self.feature_map["low_"]],
                close_=self.data[self.data_idx, self.asset_map[asset], self.feature_map["close_"]]
            )
            events.append(new_event)
        self.data_idx += 1
        return True

class HistoricalFlowHandler:
    def __init__(self, max_time=None, max_it=None):
        self.active = True
        self.start_time = time()
        self.it = 0
        self.max_time = max_time
        self.max_it = max_it
    
    def check():
        if self.max_time:
            timelapse = time() - self.start_time
            if timelapse > self.max_time:
                self.active = False
        if self.max_it:
            if self.it > self.max_it:
                self.active = False
            self.it += 1
            
    def handle_no_data():
        self.active = False

# order
@dataclass
class OrderEvent:
    timestamp: datetime
    asset: str
    direction: str # buy / sell
    quantity: int

@dataclass
class FillEvent:
    asset: str
    timestamp: datetime
    direction: str # buy / sell
    quantity: int
    fill_price: float
    commission: float = 0

# order
class Portfolio:
    def __init__(self):
        self.orders = deque()
        self.positions = List[str]
    def execute_order

# signal event
@dataclass
class SignalEvent:
    asset: str
    timestamp: datetime
    direction: str # LONG / SHORT
    
# strategies
class MeanReversionStrategy:
    def __init__(self, window_size: int, entry_thr: float, exit_thr: float):
        self.window_size = window_size
        self.entry_thr = entry_thr
        self.exit_thr = exit_thr
        
    def analyze_data(event: DataEvent, data_handler: HistoricDataHandler, events: deque, portfolio: Portfolio):
        closes = data_handler.get([event.asset], ["close"], self.window_size)
        if len(closes) < self.window_size:
            pass
        #
        else:
            z = (event.close_ - np.mean(closes)) / np.std(closes)
            if event.asset not in portfolio.positions:
                if z < -self.entry_thr:
                    events.append(
                        SignalEvent(
                            asset=event.asset,
                            timestamp=event.timestamp,
                            direction="LONG"
                        )
                    )
                elif z > self.entry_thr:
                    events.append(
                        SignalEvent(
                            asset=event.asset,
                            timestamp=event.timestamp,
                            direction="SHORT"
                        )
                    )
            #
            else:
                if abs(z) < exit_thr:
                    events.append(
                        SignalEvent(
                            asset=event.asset,
                            timestamp=event.timestamp,
                            direction="EXIT"
                        )
                    )
# main loop
class EventLoop:
    def __init__(self):
        self.events = deque()
        self.data_handler = HistoricDataHandler(TEST_DATA, ASSET_MAP, FEATURE_MAP)
        self.flow_handler = HistoricalFlowHandler()
        self.portfolio = Portfolio()
        self.active = True
    
    def start(strategy: MeanReversionStrategy):
        while self.active:
            self.flow_handler.check()
            if datahandler.update(events):
                while events:
                    event = events.popleft()
                    if isinstance(event, DataEvent):
                        strategy.analyze_data(event, self.data_handler, self.events, self.portfolio)
                    if isinstance(event, SignalEvent):
                        
            else:
                self.flow_handler.handle_no_data()
    
backtest()