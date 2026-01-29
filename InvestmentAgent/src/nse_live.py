# nse_live.py (updated key methods)

import nsepython
from datetime import datetime

class NSEStockAnalyzer:
    def __init__(self):
        pass

    def _safe_float(self, val, default=0.0):
        if val is None:
            return default
        try:
            return float(str(val).replace(',', ''))
        except:
            return default

    def get_stock_data(self, symbol: str) -> dict:
        """Returns structured JSON data for a single stock"""
        symbol = symbol.upper().strip()
        try:
            data = nsepython.nse_eq(symbol)
        except:
            try:
                data = nsepython.nse_fno(symbol)
            except:
                return {"error": f"Could not fetch data for {symbol}"}

        if 'info' not in data:
            return {"error": "Invalid response structure"}

        info = data['info']
        price = data['priceInfo']
        meta = data.get('metadata', {})
        industry = data.get('industryInfo', {})

        return {
            "symbol": info['symbol'],
            "company": info.get('companyName', 'N/A'),
            "price": self._safe_float(price['lastPrice']),
            "change": self._safe_float(price.get('change', 0)),
            "p_change": self._safe_float(price.get('pChange', 0)),
            "previous_close": self._safe_float(price['previousClose']),
            "open": self._safe_float(price['open']),
            "day_high": self._safe_float(price['intraDayHighLow']['max']),
            "day_low": self._safe_float(price['intraDayHighLow']['min']),
            "vwap": self._safe_float(price['vwap']),
            "52w_high": self._safe_float(price['weekHighLow']['max']),
            "52w_low": self._safe_float(price['weekHighLow']['min']),
            "pe": meta.get('pdSymbolPe', 'N/A'),
            "sector_pe": meta.get('pdSectorPe', 'N/A'),
            "industry": industry.get('basicIndustry', 'N/A'),
            "last_update": meta.get('lastUpdateTime', 'N/A')
        }

    def compare_stocks(self, symbols: list) -> dict:
        """Returns JSON comparison of multiple stocks"""
        results = []
        for sym in symbols:
            data = self.get_stock_data(sym)
            if "error" not in data:
                results.append(data)
            else:
                results.append({"symbol": sym.upper(), "error": "Not found"})

        return {"stocks": results, "count": len(results)}
    
    def _safe_float(self, value, default='N/A'):
        """Safely convert to float, handling string/object types"""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.replace(',', ''))
            except:
                return default
        return default

    def display_stock_summary(self, symbol: str):
        """Displays a clean summary with safe float handling"""
        try:
            data = nsepython.nse_eq(symbol.upper())
            info = data['info']
            meta = data['metadata']
            price = data['priceInfo']
            industry = data['industryInfo']
            intra = price['intraDayHighLow']
            week52 = price['weekHighLow']

            # Safe float conversion helper
            def f(val, default=0.0):
                try:
                    return float(str(val).replace(',', ''))
                except:
                    return default

            last_price = f(price['lastPrice'])
            change = f(price.get('change', 0))
            p_change = f(price.get('pChange', 0))
            prev_close = f(price['previousClose'])
            open_price = f(price['open'])
            vwap = f(price['vwap'])
            day_high = f(intra['max'])
            day_low = f(intra['min'])
            high_52 = f(week52['max'])
            low_52 = f(week52['min'])
            pe = f(meta.get('pdSymbolPe', 'N/A'))
            sector_pe = f(meta.get('pdSectorPe', 'N/A'))

            print(f"\n{'='*60}")
            print(f"         {info['symbol']} - {info['companyName']}")
            print(f"{'='*60}")
            print(f"Industry         : {industry['basicIndustry']} ({industry['sector']})")
            print(f"Current Price    : ₹{last_price:,.2f}")
            print(f"Change Today     : {change:+.2f} ({p_change:+.2f}%)")
            print(f"Previous Close   : ₹{prev_close:,.2f}")
            print(f"Open             : ₹{open_price:,.2f}")
            print(f"Day High / Low   : ₹{day_high:,.2f} / ₹{day_low:,.2f}")
            print(f"VWAP             : ₹{vwap:,.2f}")
            print(f"52-Week High     : ₹{high_52:,.2f} ({week52.get('maxDate', 'N/A')})")
            print(f"52-Week Low      : ₹{low_52:,.2f} ({week52.get('minDate', 'N/A')})")
            print(f"P/E Ratio        : {pe if pe != 'N/A' else 'N/A'}")
            print(f"Sector P/E       : {sector_pe if sector_pe != 'N/A' else 'N/A'}")
            print(f"Last Updated     : {meta.get('lastUpdateTime', 'N/A')}")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"Error fetching/displaying data for {symbol}: {str(e)}")

    def display_top_gainers(self, count: int = 10):
        """Displays top gainers with proper type handling"""
        try:
            gainers = nsepython.nse_get_top_gainers()
            print(f"\n{'='*70}")
            print(f"           TOP {count} GAINERS TODAY")
            print(f"{'='*70}")
            print(f"{'Rank':<4} {'Symbol':<10} {'Price':>12} {'% Change':>12} {'Volume (L)':>15}")
            print(f"{'-'*70}")

            for i, row in gainers.head(count).iterrows():
                price = self._safe_float(row['lastPrice'], 0)
                pchange = self._safe_float(row['pChange'], 0)
                volume = self._safe_float(row['totalTradedVolume'], 0)
                volume_lakhs = volume / 100000 if volume != 'N/A' else 'N/A'

                print(f"{i+1:<4} {row['symbol']:<10} "
                      f"₹{price:>10,.2f} {pchange:+>10.2f}% {volume_lakhs:>14,.1f}")
            print(f"{'='*70}\n")

        except Exception as e:
            print(f"Error fetching gainers: {str(e)}")

    def display_top_losers(self, count: int = 10):
        """Displays top losers with proper type handling"""
        try:
            losers = nsepython.nse_get_top_losers()
            print(f"\n{'='*70}")
            print(f"           TOP {count} LOSERS TODAY")
            print(f"{'='*70}")
            print(f"{'Rank':<4} {'Symbol':<10} {'Price':>12} {'% Change':>12} {'Volume (L)':>15}")
            print(f"{'-'*70}")

            for i, row in losers.head(count).iterrows():
                price = self._safe_float(row['lastPrice'], 0)
                pchange = self._safe_float(row['pChange'], 0)
                volume = self._safe_float(row['totalTradedVolume'], 0)
                volume_lakhs = volume / 100000 if volume != 'N/A' else 'N/A'

                print(f"{i+1:<4} {row['symbol']:<10} "
                      f"₹{price:>10,.2f} {pchange:+>10.2f}% {volume_lakhs:>14,.1f}")
            print(f"{'='*70}\n")

        except Exception as e:
            print(f"Error fetching losers: {str(e)}")

    def quick_compare(self, symbols: list):
        """Quick side-by-side comparison with reliable data parsing"""
        print(f"\n{'='*90}")
        print(f"{'QUICK COMPARISON':^90}")
        print(f"{'='*90}")
        print(f"{'Symbol':<10} {'Price':>14} {'% Chg':>12} {'Day Range':>22} {'Volume (L)':>14}")
        print(f"{'-'*90}")

        for sym in symbols:
            try:
                data = nsepython.nse_eq(sym.upper())
                info = data['info']
                price = data['priceInfo']
                intra = price['intraDayHighLow']
                meta = data.get('metadata', {})

                # Safe float conversion
                def f(val, default='N/A'):
                    try:
                        return float(str(val).replace(',', ''))
                    except:
                        return default

                last_price = f(price['lastPrice'])
                p_change = f(price['pChange'])
                day_high = f(intra['max'])
                day_low = f(intra['min'])

                # Volume fallback chain
                vol = meta.get('totalTradedVolume', 'N/A')
                if vol == 'N/A':
                    vol = price.get('totalTradedVolume', 'N/A')
                vol_lakhs = f(vol) / 100000 if f(vol) != 'N/A' else 'N/A'
                vol_str = f"{vol_lakhs:,.1f}" if vol_lakhs != 'N/A' else 'N/A'

                print(f"{info['symbol']:<10} "
                    f"₹{last_price:>12,.2f} "
                    f"{p_change:+>10.2f}% "
                    f"₹{day_high:>8,.0f} - ₹{day_low:<8,.0f} "
                    f"{vol_str:>12}")

            except Exception as e:
                print(f"{sym.upper():<10} {'Data Error':>70}")

        print(f"{'='*90}\n")


# # === Example Usage ===
if __name__ == "__main__":
    analyzer = NSEStockAnalyzer()

    print("\n" + "="*60)
    print("           NSE LIVE MARKET DASHBOARD")
    print("="*60)

    # Stock Summary
    #analyzer.display_stock_summary('LT')

    # Market Movers
    analyzer.display_top_gainers(10)
    analyzer.display_top_losers(10)

    # Quick Comparison
    analyzer.quick_compare(['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'LT', 'ITC', 'SBIN'])