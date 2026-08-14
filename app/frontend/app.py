import os
import streamlit as st
import pandas as pd
import psycopg2
import requests
from dotenv import load_dotenv

# Importamos las funciones de visualización desde nuestro módulo
from visualizaciones import (
    plot_top_metric_bar,
    plot_titularidad_vs_produccion,
    plot_perfil_percentiles_jugador,
    plot_probabilidad_atribucion_donut
)

# Cargar variables de entorno
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Madrid rio-Segunda juvenil - Analytics",
    page_icon="⚽",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. CARGA DE DATOS DESDE POSTGRESQL (CON CACHÉ)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def obtener_datos_jugadores() -> pd.DataFrame:
    """Consulta la base de datos PostgreSQL y devuelve un DataFrame con las métricas."""
    try:
        with psycopg2.connect(DATABASE_URL, client_encoding='utf8') as conn:
            df = pd.read_sql_query("SELECT * FROM jugadores_stats;", conn)
        return df
    except Exception as e:
        st.error(f"Error al conectar con la base de datos PostgreSQL: {e}")
        return pd.DataFrame()

# Cargar el DataFrame principal
df = obtener_datos_jugadores()

# -----------------------------------------------------------------------------
# 👈 BARRA LATERAL (SIDEBAR)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚽ Madrid rio-Segunda juvenil-Analytics")
    st.markdown("---")
    
    # Tabla resumida del equipo: Métricas globales
    st.subheader("📊 Métricas Globales")
    if not df.empty:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Jugadores", len(df))
            st.metric("Goles Totales", int(df['goles'].sum()) if 'goles' in df.columns else 0)
        with col_m2:
            st.metric("Partidos", int(df['partidos_jugados'].max()) if 'partidos_jugados' in df.columns else 0)
            st.metric("Asist. Totales", int(df['asist'].sum()) if 'asist' in df.columns else 0)
            
        st.markdown("---")
        
        # Selector de Jugador (Conectado dinámicamente al Bloque 3)
        st.subheader("🔍 Buscador de Jugador")
        jugadores_lista = sorted(df['nombre'].unique())
        jugador_seleccionado = st.selectbox(
            "Selecciona un jugador para ver su perfil:",
            options=jugadores_lista
        )
    else:
        st.warning("No se pudieron cargar los datos desde la BD.")

# -----------------------------------------------------------------------------
# 🎯 ÁREA PRINCIPAL (4 BLOQUES CONSECUTIVOS)
# -----------------------------------------------------------------------------
st.title("🏆 Dashboard de Rendimiento del Plantel")
st.markdown("Análisis interactivo de estadísticas individuales, rendimiento ofensivo y modelos predictivos.")
st.markdown("---")

if not df.empty:

    # =========================================================================
    # BLOQUE 1: TOP 5 DEL EQUIPO (UNIFICADO EN PESTAÑAS)
    # =========================================================================
    st.header("1. Top 5 del Equipo")
    
    tab1, tab2, tab3 = st.tabs(["⚽ Top Goleadores", "🅰️ Top Asistidores", "🔥 Contribución x Partido"])
    
    with tab1:
        fig_goles = plot_top_metric_bar(
            df, 
            metric_col='goles', 
            title="Top 5 Goleadores", 
            x_axis_label="Goles Totales", 
            color_hex="#426a09"
        )
        st.plotly_chart(fig_goles, use_container_width=True)
        
    with tab2:
        fig_asist = plot_top_metric_bar(
            df, 
            metric_col='asist', 
            title="Top 5 Asistidores", 
            x_axis_label="Asistencias Totales", 
            color_hex="#090f6a"
        )
        st.plotly_chart(fig_asist, use_container_width=True)
        
    with tab3:
        fig_contrib = plot_top_metric_bar(
            df, 
            metric_col='contribucion_gol', 
            title="Top 5 Contribución de Gol x Partido", 
            x_axis_label="Contribución de Gol (G+A / PJ)", 
            color_hex="#bfc20d"
        )
        st.plotly_chart(fig_contrib, use_container_width=True)

    st.markdown("---")

    # =========================================================================
    # BLOQUE 2: MATRIZ TITULARIDAD VS. PRODUCCIÓN OFENSIVA
    # =========================================================================
    st.header("2. Matriz: Titularidad vs. Producción Ofensiva")
    
    fig_matriz = plot_titularidad_vs_produccion(
        df,
        x_col='pct_titularidad',
        y_col='contribucion_gol',
        name_col='nombre'
    )
    st.plotly_chart(fig_matriz, use_container_width=True)

    st.markdown("---")

    # =========================================================================
    # BLOQUE 3: PERFIL DE IMPACTO INDIVIDUAL EN PERCENTILES
    # =========================================================================
    st.header("3. Perfil de Impacto Individual")
    
    if jugador_seleccionado:
        fig_percentiles = plot_perfil_percentiles_jugador(
            df=df,
            nombre_jugador=jugador_seleccionado
        )
        if fig_percentiles:
            st.plotly_chart(fig_percentiles, use_container_width=True)
        else:
            st.error("No se pudo generar el perfil para el jugador seleccionado.")

    st.markdown("---")

    # =========================================================================
    # BLOQUE 4: MODELO PREDICTIVO DE ATRIBUCIÓN POR POSICIÓN
    # =========================================================================
    st.header("4. Modelo Predictivo: Atribución por Posición (Naive Bayes)")
    
    col_pred_1, col_pred_2 = st.columns([1, 2])
    
    with col_pred_1:
        st.subheader("Configuración del Evento")
        evento_seleccionado = st.radio(
            "Selecciona el tipo de evento a analizar:",
            options=["Gol", "Asistencia"],
            help="El modelo Naive Bayes calculará la probabilidad de participación por línea posicional."
        )
        
    with col_pred_2:
        try:
            # Petición HTTP al endpoint de FastAPI
            response = requests.get(
                f"{API_URL}/prediccion/atribucion", 
                params={"tipo_evento": evento_seleccionado},
                timeout=5
            )
            
            if response.status_code == 200:
                data_pred = response.json()
                fig_donut = plot_probabilidad_atribucion_donut(
                    distribucion_data=data_pred["distribucion_probabilidad"],
                    evento=data_pred["evento"]
                )
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.error("Error al consultar el servicio predictivo de FastAPI.")
                
        except requests.exceptions.ConnectionError:
            st.warning("⚠️ No se puede conectar con la API (FastAPI). Comprueba que Uvicorn esté corriendo en la terminal.")