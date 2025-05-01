import ccxt
import pandas as pd
import time
from datetime import datetime
import numpy as np
from collections import deque
import threading
import logging
import os
import csv

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("pandas_ta 库未安装，将使用自定义RSI计算方法")
    print("建议安装 pandas_ta: pip install pandas-ta")

# 配置日志
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            f"{log_dir}/rsi_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)


class HighFrequencyRSI:
    def __init__(self, period=14, overbought=70, oversold=30):
        self.exchange = ccxt.binance({
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
            },
            'enableRateLimit': True,  # 启用ccxt的内置限速
        })

        self.symbol = 'BTC/USDT'
        self.timeframe = '15m'
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

        # 用于限速的变量
        self.request_count = 0
        self.last_request_time = time.time()
        self.request_limit = 1800  # 每分钟请求限制
        self.request_window = 60   # 60秒窗口
        self.min_interval = 0.1    # 最小请求间隔100ms

        # 数据缓存
        self.price_cache = deque(maxlen=100)
        self.last_price = None
        self.last_rsi = None

        # 用于控制程序终止
        self.is_running = True

        # 数据保存
        self.data_dir = "data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.csv_file = f"{self.data_dir}/rsi_data_{self.period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # 创建CSV文件标题
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'price', 'rsi', 'market_status'])

    def calculate_rsi_pandas(self, closes, period=14):
        """使用纯pandas计算RSI指标"""
        # 确保closes是pandas.Series
        closes = pd.Series(closes)

        # 计算价格变化
        delta = closes.diff()

        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # 计算平均涨幅和平均跌幅
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # 计算相对强度RS
        rs = avg_gain / avg_loss

        # 计算RSI
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_rsi(self, df, period=14):
        """计算RSI，优先使用pandas_ta，如果不可用则使用自定义方法"""
        if PANDAS_TA_AVAILABLE:
            # 使用pandas_ta库计算RSI
            return ta.rsi(df['close'], length=period)
        else:
            # 使用自定义方法计算RSI
            return self.calculate_rsi_pandas(df['close'], period)

    def calculate_multiple_rsi(self, df):
        """计算多个周期的RSI用于验证"""
        periods = [10, 14, 20]
        results = {}

        for p in periods:
            if p == self.period:
                continue
            results[p] = self.calculate_rsi(df, p).iloc[-1]

        return results

    def can_make_request(self):
        """检查是否可以发送新请求"""
        current_time = time.time()
        if current_time - self.last_request_time < self.min_interval:
            return False

        # 重置计数器
        if current_time - self.last_request_time >= self.request_window:
            self.request_count = 0
            self.last_request_time = current_time

        return self.request_count < self.request_limit

    def get_market_status(self, rsi):
        """根据RSI值判断市场状态"""
        if rsi >= self.overbought:
            return "超买", "\033[91m"  # 红色
        elif rsi <= self.oversold:
            return "超卖", "\033[92m"  # 绿色
        else:
            return "正常", "\033[0m"   # 默认颜色

    def save_data(self, timestamp, price, rsi, status):
        """保存数据到CSV文件"""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, price, rsi, status])

    def fetch_and_process(self):
        """获取并处理数据"""
        try:
            if not self.can_make_request():
                time.sleep(self.min_interval)
                return

            ohlcv = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=100
            )

            self.request_count += 1
            self.last_request_time = time.time()

            if not ohlcv:
                return

            df = pd.DataFrame(
                ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # 转换时间戳为日期时间
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            current_price = df['close'].iloc[-1]

            # 计算RSI
            rsi_values = self.calculate_rsi(df, self.period)
            current_rsi = rsi_values.iloc[-1]

            # 计算其他周期的RSI用于交叉验证
            other_rsi = self.calculate_multiple_rsi(df)

            # 计算价格变化
            price_change = 0
            if self.last_price:
                price_change = (
                    (current_price - self.last_price) / self.last_price) * 100

            # 获取市场状态
            status, color = self.get_market_status(current_rsi)

            # 保存数据
            self.save_data(
                df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S'),
                current_price,
                current_rsi,
                status
            )

            # 记录日志
            logging.info(
                f"价格: {current_price:.2f}, RSI({self.period}): {current_rsi:.2f}, 状态: {status}")

            # 清屏并显示数据
            print("\033[H\033[J")  # 清屏
            print(
                f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
            print(f"当前价格: {current_price:.2f} USDT")
            print(f"价格变化: {price_change:+.2f}%")
            print(f"RSI({self.period}): {color}{current_rsi:.2f}\033[0m")
            print(f"市场状态: {color}{status}\033[0m")

            # 显示其他周期的RSI用于验证
            print("\n交叉验证:")
            for p, v in other_rsi.items():
                print(f"RSI({p}): {v:.2f}")

            print(f"\n更新频率: {self.min_interval * 1000:.0f}ms")
            print(f"K线时间: {df['timestamp'].iloc[-1]}")
            library_info = "pandas_ta" if PANDAS_TA_AVAILABLE else "自定义pandas方法"
            print(f"RSI计算方法: {library_info}")
            print(f"数据记录: {self.csv_file}")
            print("按 Ctrl+C 停止程序")

            self.last_price = current_price
            self.last_rsi = current_rsi

        except ccxt.NetworkError as e:
            logging.error(f"网络错误: {e}")
        except Exception as e:
            logging.error(f"发生错误: {e}")

    def run(self):
        """主运行循环"""
        print(f"开始高频监控 {self.symbol} 永续合约...")
        library_info = "pandas_ta" if PANDAS_TA_AVAILABLE else "自定义pandas方法"
        print(f"使用{library_info}计算{self.period}周期RSI...")
        print(f"数据保存路径: {self.csv_file}")
        print("正在初始化数据...")

        try:
            while self.is_running:
                self.fetch_and_process()
                time.sleep(self.min_interval)
        except KeyboardInterrupt:
            self.is_running = False
            print("\n程序已停止")
            print(f"数据已保存到: {self.csv_file}")


def main():
    # 可以通过参数配置RSI设置
    monitor = HighFrequencyRSI(
        period=20,        # RSI周期
        overbought=10,    # 超买阈值
        oversold=10       # 超卖阈值
    )
    monitor.run()


if __name__ == "__main__":
    main()
