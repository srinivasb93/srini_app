from unittest.mock import MagicMock


class MockUpstoxApi:
    class OrderApi:
        def place_order(self, order, api_version="v2"):
            return MagicMock(data=MagicMock(order_id="MOCK12345"))
        def get_order_status(self, order_id):
            return MagicMock(data=MagicMock(status="complete"))
        def get_order_book(self, api_version="v2"):
            return MagicMock(data=[MagicMock(to_dict=lambda: {"order_id": "MOCK12345", "trading_symbol": "INFY"})])

    class MarketQuoteApi:
        def get_full_market_quote(self, instruments, api_version="v2"):
            return MagicMock(data={
                "NSE:INFY": MagicMock(to_dict=lambda: {
                    "last_price": 1500.0,
                    "volume": 1000,
                    "ohlc": {"open": 1490, "high": 1510, "low": 1480, "close": 1500}
                })
            })

class MockZerodhaApi:
    def place_order(self, variety, **params):
        return "Z12345"
    def order_history(self, order_id):
        return [{"status": "COMPLETE"}]
    def orders(self):
        return [{"order_id": "Z12345", "tradingsymbol": "INFY"}]
    def mf_sip(self, fund, amount, frequency, start_date, instalments):
        return {"sip_id": "SIP123"}
    def cancel_mf_sip(self, sip_id):
        return True