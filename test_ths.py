import akshare as ak
try:
    df = ak.stock_board_industry_name_ths()
    print("THS columns:", df.columns)
    print(df.head(2))
except Exception as e:
    print("THS Error:", e)

try:
    df = ak.stock_board_concept_name_ths()
    print("THS Concept columns:", df.columns)
except Exception as e:
    print("THS Concept Error:", e)
