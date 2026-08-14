import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def obtener_datos_jugadores() -> pd.DataFrame:
    """Consulta la base de datos PostgreSQL y devuelve un DataFrame con las métricas."""
    with psycopg2.connect(DATABASE_URL, client_encoding='utf8') as conn:
        df = pd.read_sql_query("SELECT * FROM jugadores_stats;", conn)
    return df