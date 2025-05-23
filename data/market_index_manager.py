import pandas as pd

def get_index(index_name, timestamp):
    from data_manager import get_connection  # import داخل تابع
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return 0.0
        query = """
            SELECT value FROM market_indices
            WHERE index_name = %s AND timestamp <= %s
            ORDER BY timestamp DESC LIMIT 1
        """
        df = pd.read_sql(query, con=conn, params=(index_name, timestamp))
        return df['value'].iloc[0] if not df.empty else 0.0
    except Exception as e:
        print(f"Error fetching index {index_name}: {e}")
        return 0.0
    finally:
        if conn:
            conn.close()
            