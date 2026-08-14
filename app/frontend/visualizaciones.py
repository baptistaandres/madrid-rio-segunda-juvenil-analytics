import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

def plot_top_metric_bar(
    df: pd.DataFrame, 
    metric_col: str, 
    title: str, 
    x_axis_label: str, 
    color_hex: str, 
    top_n: int = 5
):
    """
    Genera un gráfico de barras horizontal parametrizado para métricas Top N de jugadores.
    """
    top_df = df.nlargest(top_n, metric_col).sort_values(metric_col)
    
    fig = px.bar(
        top_df,
        x=metric_col,
        y='nombre',
        orientation='h',
        title=title,
        labels={metric_col: x_axis_label, 'nombre': 'Jugador'},
        color_discrete_sequence=[color_hex],
        text=metric_col
    )
    
    # 3. Personalizar el diseño
    fig.update_layout(
        xaxis_title=x_axis_label,
        yaxis_title="Jugador",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    ) 
    
    fig.update_traces(textposition='outside')
    
    return fig


def plot_titularidad_vs_produccion(
    df: pd.DataFrame, 
    x_col: str = 'pct_titularidad', 
    y_col: str = 'contribucion_gol',
    name_col: str = 'nombre'
):
    """
    Genera un Scatter Plot de Titularidad vs Producción Ofensiva 
    con líneas de referencia en las medias para análisis de cuadrantes.
    """
    x_mean = df[x_col].mean()
    y_mean = df[y_col].mean()

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        text=name_col,
        title="Matriz: % Titularidad vs. Producción Ofensiva",
        labels={
            x_col: '% Titularidad',
            y_col: 'Contribución de Gol X Partido',
            name_col: 'Jugador'
        },
        hover_data=[name_col, x_col, y_col]
    )

    fig.update_traces(
        textposition='top center',
        marker=dict(size=10, color='#090f6a')
    )

    fig.add_vline(
        x=x_mean, 
        line_dash="dash", 
        line_color="gray", 
        annotation_text=f"Media Titularidad ({x_mean:.1f}%)", 
        annotation_position="top left"
    )
    fig.add_hline(
        y=y_mean, 
        line_dash="dash", 
        line_color="gray", 
        annotation_text=f"Media Contribución ({y_mean:.2f})", 
        annotation_position="bottom right"
    )

    fig.update_layout(
        xaxis_title="% Titularidad",
        yaxis_title="Contribución de Gol X Partido",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig

def plot_perfil_percentiles_jugador(
    df: pd.DataFrame, 
    nombre_jugador: str,
    metricas: list = None,
    etiquetas: list = None
) -> go.Figure:
    """
    Genera el gráfico de Lollipop Horizontal con los percentiles y datos reales
    de un jugador seleccionado por su nombre.
    """
    if metricas is None:
        metricas = ['gol_x_part', 'asist_x_part', 'contribucion_gol', 'pct_titularidad']
    if etiquetas is None:
        etiquetas = ['Goles / Partido', 'Asistencias / Partido', 'Contribución Gol', '% Titularidad']

    match_jugador = df[df['nombre'] == nombre_jugador]
    if match_jugador.empty:
        return None 

    idx = match_jugador.index[0]


    df_percentiles = df[metricas].rank(pct=True) * 100

    valores_pct = df_percentiles.loc[idx, metricas].values
    valores_reales = df.loc[idx, metricas].values

    fig = go.Figure()

    # Líneas de la barra (Lollipops)
    for i in range(len(metricas)):
        fig.add_trace(go.Scatter(
            x=[0, valores_pct[i]],
            y=[etiquetas[i], etiquetas[i]],
            mode='lines',
            line=dict(color='#2b5c8f', width=4),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.add_trace(go.Scatter(
        x=valores_pct,
        y=etiquetas,
        mode='markers+text',
        marker=dict(color='#2b5c8f', size=16),
        text=[f"{v:.0f}%" for v in valores_pct],
        textposition="top center",
        hovertemplate="<b>%{y}</b><br>Percentil: %{x:.1f}%<br>Dato Real: %{customdata}<extra></extra>",
        customdata=valores_reales,
        name=nombre_jugador
    ))

    fig.add_vline(
        x=50, 
        line_dash="dash", 
        line_color="gray", 
        annotation_text="Media del equipo (P50)"
    )

    fig.update_layout(
        title=f"Perfil de Impacto en Percentiles: <b>{nombre_jugador}</b>",
        xaxis=dict(title="Percentil en la Plantilla (0 = Mínimo, 100 = Máximo)", range=[0, 105]),
        yaxis=dict(title="Métricas"),
        showlegend=False,
        height=450,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig

def plot_probabilidad_atribucion_donut(distribucion_data: list, evento: str):
    """
    Recibe la lista de diccionarios que devuelve el endpoint de FastAPI
    y genera un gráfico de Donut de las probabilidades por posición.
    """
    # Convertir el JSON devuelto por la API en un DataFrame para Plotly
    df_probs = pd.DataFrame(distribucion_data)

    fig = px.pie(
        df_probs,
        values='probabilidad_pct',
        names='posicion',
        title=f"Atribución Probabilística por Posición para: <b>{evento}</b>",
        hole=0.4,  # Transforma el gráfico de pastel en Donut
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_traces(
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>Probabilidad: %{value:.2f}%<extra></extra>"
    )

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True
    )

    return fig
