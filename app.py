"""
app.py
Dashboard hecho con Streamlit. Se corre con: streamlit run app.py
Se sube un video y se ve procesado con el conteo y la congestion al lado.
"""

import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from analizador import AnalizadorTrafico, CLASES_VEHICULO

st.set_page_config(page_title="Analizador de tráfico", page_icon="🚦", layout="wide")

st.title("🚦 Analizador inteligente de tráfico")
st.caption("Detección y conteo de vehículos con YOLO + Computer Vision")

# barra lateral con los parametros
with st.sidebar:
    st.header("Configuración")
    modelo = st.selectbox(
        "Modelo YOLO",
        ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
        help="n = más rápido, m = más preciso. Con CPU usa la versión n.",
    )
    confianza = st.slider("Confianza mínima", 0.1, 0.9, 0.35, 0.05,
                          help="Detecciones con menor confianza se descartan.")
    linea = st.slider("Posición de la línea de conteo", 0.1, 0.9, 0.6, 0.05,
                      help="Fracción del alto del video (0 = arriba, 1 = abajo).")
    capacidad = st.slider("Capacidad de la vía (vehículos en pantalla)", 5, 80, 25,
                          help="Cuántos vehículos visibles consideramos 'vía llena'.")
    saltar = st.slider("Procesar 1 de cada N frames", 1, 4, 1,
                       help="Súbelo si tu computador va lento.")
    st.divider()
    archivo = st.file_uploader("Video de la intersección", type=["mp4", "avi", "mov", "mkv"])
    ruta_local = st.text_input("…o ruta de un video en tu PC", "")
    iniciar = st.button("▶ Procesar video", type="primary", width="stretch")


@st.cache_resource
def cargar_modelo(nombre):
    # cache_resource es para no cargar YOLO otra vez cada vez que se toca algo
    return AnalizadorTrafico(modelo=nombre)


# video a la izquierda y estadisticas a la derecha
col_video, col_stats = st.columns([3, 2])
with col_video:
    hueco_video = st.empty()
    barra = st.progress(0.0, text="Esperando video…")
with col_stats:
    hueco_nivel = st.empty()
    m1, m2, m3 = st.columns(3)
    hueco_total, hueco_pantalla, hueco_vpm = m1.empty(), m2.empty(), m3.empty()
    st.markdown("**Vehículos contados por tipo**")
    hueco_barras = st.empty()
    st.markdown("**Vehículos en pantalla e índice de congestión**")
    hueco_linea = st.empty()

COLOR_HTML = {"Bajo": "#2e9e4f", "Medio": "#e0a800", "Alto": "#d63b3b"}


def pintar_estadisticas(est, historial):
    nivel = est["nivel_congestion"]
    hueco_nivel.markdown(
        f"<div style='padding:14px;border-radius:10px;background:{COLOR_HTML[nivel]};"
        f"color:white;font-size:22px;text-align:center'>"
        f"Congestión: <b>{nivel}</b> &nbsp;·&nbsp; índice {est['indice_congestion']:.2f}</div>",
        unsafe_allow_html=True,
    )
    hueco_total.metric("Total contados", est["total_contados"])
    hueco_pantalla.metric("En pantalla", est["en_pantalla"])
    vpm = est["total_contados"] / max(est["tiempo_s"], 1) * 60
    hueco_vpm.metric("Vehículos/min", f"{vpm:.1f}")

    df_clases = pd.DataFrame(
        {"Vehículos": list(est["contados_por_clase"].values())},
        index=list(est["contados_por_clase"].keys()),
    )
    hueco_barras.bar_chart(df_clases, height=220)

    if historial:
        df = pd.DataFrame(historial).set_index("segundo")
        hueco_linea.line_chart(df[["en_pantalla", "indice_x10"]], height=220)


if iniciar:
    # si subieron el video se guarda en un archivo temporal
    if archivo is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(archivo.name).suffix)
        tmp.write(archivo.read())
        tmp.close()
        ruta_video = tmp.name
    elif ruta_local.strip():
        ruta_video = ruta_local.strip()
    else:
        st.warning("Sube un video o escribe su ruta.")
        st.stop()

    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        st.error("No se pudo abrir el video.")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    analizador = cargar_modelo(modelo)
    analizador.confianza = confianza
    analizador.posicion_linea = linea
    analizador.capacidad = capacidad
    analizador.fps = fps / saltar
    analizador.reiniciar()

    historial = []
    leidos = 0
    inicio = time.time()
    est = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        leidos += 1
        if leidos % saltar:
            continue

        anotado, est = analizador.procesar_frame(frame)
        hueco_video.image(cv2.cvtColor(anotado, cv2.COLOR_BGR2RGB), width="stretch")

        # las graficas se actualizan 2 veces por segundo, si no la app va muy lenta
        if est["frame"] % max(int(analizador.fps / 2), 1) == 0:
            historial.append({
                "segundo": est["tiempo_s"],
                "en_pantalla": est["en_pantalla"],
                "indice_x10": est["indice_congestion"] * 10,
            })
            pintar_estadisticas(est, historial)
            ritmo = est["frame"] / (time.time() - inicio)
            barra.progress(min(leidos / total, 1.0),
                           text=f"Procesando… {leidos}/{total} frames · {ritmo:.1f} fps")

    cap.release()
    if est is None:
        st.error("El video no tenía frames.")
        st.stop()

    pintar_estadisticas(est, historial)
    barra.progress(1.0, text="✅ Video procesado")

    st.subheader("Resumen")
    resumen = pd.DataFrame({
        "Tipo": list(CLASES_VEHICULO.values()),
        "Cruzaron la línea": list(est["contados_por_clase"].values()),
        "Detectados (IDs únicos)": list(est["unicos_por_clase"].values()),
    })
    st.dataframe(resumen, hide_index=True, width="stretch")
    st.write(f"Dirección — bajando: **{est['direccion']['bajando']}**, "
             f"subiendo: **{est['direccion']['subiendo']}**")
    st.download_button(
        "⬇ Descargar estadísticas (CSV)",
        pd.DataFrame(historial).to_csv(index=False).encode("utf-8"),
        "estadisticas_trafico.csv",
        "text/csv",
    )
else:
    with col_video:
        st.info("Sube un video en la barra lateral y presiona **Procesar video**.")
