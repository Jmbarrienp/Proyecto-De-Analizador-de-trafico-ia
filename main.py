"""
main.py
Procesa un video desde la terminal y guarda en la carpeta resultados el video dibujado,
un csv con los datos de cada segundo y un json con el resumen.

Ejemplos:
    python main.py --video videos/2_autopista_con_camiones.mp4
    python main.py --video videos/1_interseccion_semaforo_1.mp4 --linea 0.5
    python main.py --video 0 --mostrar   (con la camara del computador)
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2

from analizador import AnalizadorTrafico


def main():
    p = argparse.ArgumentParser(description="Analizador inteligente de tráfico con YOLO")
    p.add_argument("--video", required=True, help="Ruta del video, o 0 para la cámara web")
    p.add_argument("--modelo", default="yolo11n.pt", help="Pesos de YOLO (yolo11n.pt, yolo11s.pt, ...)")
    p.add_argument("--conf", type=float, default=0.35, help="Confianza mínima (0-1)")
    p.add_argument("--linea", type=float, default=0.6, help="Altura de la línea de conteo (0-1)")
    p.add_argument("--capacidad", type=int, default=25, help="Vehículos en pantalla = 100 %% de densidad")
    p.add_argument("--salida", default="resultados", help="Carpeta de salida")
    p.add_argument("--mostrar", action="store_true", help="Mostrar el video mientras se procesa")
    p.add_argument("--max-segundos", type=float, default=None, help="Procesar solo los primeros N segundos")
    args = p.parse_args()

    fuente = 0 if args.video == "0" else args.video
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    analizador = AnalizadorTrafico(
        modelo=args.modelo, confianza=args.conf, posicion_linea=args.linea,
        capacidad=args.capacidad, fps=fps,
    )

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    nombre = "camara" if fuente == 0 else Path(args.video).stem
    ruta_video = salida / f"{nombre}_anotado.mp4"
    ruta_csv = salida / f"{nombre}_estadisticas.csv"
    ruta_json = salida / f"{nombre}_resumen.json"

    escritor = cv2.VideoWriter(str(ruta_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto))

    archivo_csv = open(ruta_csv, "w", newline="", encoding="utf-8")
    columnas = ["segundo", "en_pantalla", "total_contados", "Automovil", "Motocicleta", "Bus",
                "Camion", "velocidad_media", "indice_congestion", "nivel_congestion"]
    csv_writer = csv.DictWriter(archivo_csv, fieldnames=columnas)
    csv_writer.writeheader()

    print(f"Procesando {args.video}  ({ancho}x{alto} @ {fps:.1f} fps, {total_frames} frames)")
    inicio = time.time()
    est = None
    max_frames = int(args.max_segundos * fps) if args.max_segundos else None
    frames_por_segundo = max(int(round(fps)), 1)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        anotado, est = analizador.procesar_frame(frame)
        escritor.write(anotado)

        # una fila del csv por cada segundo de video
        if est["frame"] % frames_por_segundo == 0:
            fila = {
                "segundo": int(est["frame"] / frames_por_segundo),
                "en_pantalla": est["en_pantalla"],
                "total_contados": est["total_contados"],
                **est["contados_por_clase"],
                "velocidad_media": est["velocidad_media"],
                "indice_congestion": est["indice_congestion"],
                "nivel_congestion": est["nivel_congestion"],
            }
            csv_writer.writerow(fila)

        if est["frame"] % 50 == 0:
            ritmo = est["frame"] / (time.time() - inicio)
            print(f"  frame {est['frame']}/{total_frames}  ({ritmo:.1f} fps)  "
                  f"contados={est['total_contados']}  congestión={est['nivel_congestion']}")

        if args.mostrar:
            cv2.imshow("Analizador de trafico (Q para salir)", anotado)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if max_frames and est["frame"] >= max_frames:
            break

    cap.release()
    escritor.release()
    archivo_csv.close()
    cv2.destroyAllWindows()

    if est is None:
        raise SystemExit("El video no tenía frames.")

    duracion = time.time() - inicio
    resumen = {
        "video": str(args.video),
        "segundos_de_video": est["tiempo_s"],
        "segundos_de_procesamiento": round(duracion, 1),
        "fps_procesamiento": round(est["frame"] / duracion, 1),
        "total_contados": est["total_contados"],
        "contados_por_clase": est["contados_por_clase"],
        "direccion": est["direccion"],
        "vehiculos_unicos_detectados": est["unicos_por_clase"],
        "vehiculos_por_minuto": round(est["total_contados"] / max(est["tiempo_s"], 1) * 60, 1),
        "nivel_congestion_final": est["nivel_congestion"],
    }
    ruta_json.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n===== RESUMEN =====")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"\nArchivos guardados en: {salida.resolve()}")


if __name__ == "__main__":
    main()
