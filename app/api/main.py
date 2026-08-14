from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from app.api.database import obtener_datos_jugadores

app = FastAPI(
    title="Madrid Río-Segunda Juvenil Analytics API",
    description="API para consultar estadísticas e inferencia probabilística del equipo.",
    version="1.0.0"
)

@app.get("/")
def health_check():
    """Endpoint básico para verificar que la API está encendida."""
    return {"status": "ok", "message": "API Madrid Río activa y funcionando"}


@app.get("/jugador/{nombre}")
def obtener_jugador(nombre: str):
    """Devuelve las estadísticas individuales de un jugador específico."""
    df = obtener_datos_jugadores()
    
    # Búsqueda insensible a mayúsculas/minúsculas
    jugador = df[df['nombre'].str.lower() == nombre.lower()]
    
    if jugador.empty:
        raise HTTPException(
            status_code=404, 
            detail=f"Jugador '{nombre}' no encontrado en la base de datos"
        )
        
    return jugador.to_dict(orient="records")[0]

@app.get("/prediccion/atribucion")
def predecir_atribucion(tipo_evento: str = Query("Gol", description="Tipo de evento: 'Gol' o 'Asistencia'")):
    """
    Ejecuta el modelo Naive Bayes y devuelve la distribución de probabilidad
    de participación por línea posicional según el tipo de evento.
    """
    evento_formateado = tipo_evento.capitalize()
    if evento_formateado not in ["Gol", "Asistencia"]:
        raise HTTPException(
            status_code=400, 
            detail="El parámetro 'tipo_evento' debe ser 'Gol' o 'Asistencia'"
        )
        
    df = obtener_datos_jugadores()
    
    # Resampling de eventos
    goles_df = df[df['goles'] > 0].loc[df.index.repeat(df['goles'])].copy()
    goles_df['tipo_evento'] = 'Gol'

    asist_df = df[df['asist'] > 0].loc[df.index.repeat(df['asist'])].copy()
    asist_df['tipo_evento'] = 'Asistencia'

    df_eventos = pd.concat([goles_df, asist_df], ignore_index=True)

    # Entrenamiento del modelo Naive Bayes
    le_evento = LabelEncoder()
    le_posicion = LabelEncoder()

    X_eventos = le_evento.fit_transform(df_eventos['tipo_evento']).reshape(-1, 1)
    y_posicion = le_posicion.fit_transform(df_eventos['posicion'])

    modelo = MultinomialNB()
    modelo.fit(X_eventos, y_posicion)

    # Inferencia
    idx_evento = le_evento.transform([evento_formateado]).reshape(-1, 1)
    probs = modelo.predict_proba(idx_evento)[0]

    distribucion = [
        {
            "posicion": pos, 
            "probabilidad_pct": round(float(prob) * 100, 2)
        }
        for pos, prob in zip(le_posicion.classes_, probs)
    ]
    
    # Ordenar de mayor a menor probabilidad
    distribucion = sorted(distribucion, key=lambda x: x["probabilidad_pct"], reverse=True)

    return {
        "evento": evento_formateado,
        "distribucion_probabilidad": distribucion
    }

