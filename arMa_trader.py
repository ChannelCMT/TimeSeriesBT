from vector import portfolio
import pandas as pd
from datetime import datetime, timedelta

class Trader(portfolio.Portfolio):
    def init(self, symbols: list):
        super().init(symbols)
        self.order_ids = {}
        for symbol in symbols:
            self.order_ids[symbol] = ""
        self.symbols = symbols
        self.init_cash = 100000
                    
    def algorithm(self, s: dict):
        sig_ar = list(s["absorption"].values())[0]
        for symbol in self.symbols:
            signal = s[symbol]["maDiff"]
            # print(signal)
            price = s[symbol]['close']
            time = s['datetime']
            if not self.order_ids[symbol]:
                if sig_ar > 0.94:
                    print('high correlation', time, sig_ar)
                    if signal == -1:
                        print('entry short pos', symbol,  round(-1*self.init_cash/price))
                        order_id = self.entry_order(symbol, round(-1*self.init_cash/price, 2))
                        self.order_ids[symbol] = order_id
                        self.set_trailing_stop(order_id, 0.2, s[symbol]["low"])
                        # self.set_autoexit(order_id, 0.2, 0.4)
                         # 订单出场设置
                # elif   sig_overall > 4:
                #     if signal == 1:
                #         order_id = self.entry_order(symbol,round(self.init_cash/price, 2))
                #         self.order_ids.add(order_id)
            elif self.order_ids[symbol]:
                if signal==0:
                    print('closeOrder', self.order_ids[symbol])
                    op = self.get_order(self.order_ids[symbol])
                    self.exit_order(op)

        #     """
        #         有开仓时对所有的订单做混合止盈止损，幅度都为5%。
        #     """
        #     self.mix_autoexit(s, list(self.order_ids), 0.2, 0.5)
    
    def on_order(self, order: portfolio.Order):
    #     if order.status == portfolio.OrderStatus.Holding:
    #         """
    #             时间出场设置
    #         """
    #         self.timestop(order.orderId, order.entryDt + timedelta(hours=90))
        if order.status == portfolio.OrderStatus.Finished:
            self.order_ids.discard(order.orderId)

    def on_order(self, order: portfolio.Order):
        if order.status == portfolio.OrderStatus.Finished:
            self.order_ids[order.symbol] = ""