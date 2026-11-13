"""
analizador.py
Aqui esta toda la logica: detectar los vehiculos con YOLO, seguirlos con ByteTrack,
contar los que cruzan la linea, calcular la congestion y dibujar todo en el video.
"""

from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# clases de COCO que son vehiculos (YOLO viene entrenado con COCO)
CLASES_VEHICULO = {
    2: "Automovil",
    3: "Motocicleta",
    5: "Bus",
    7: "Camion",
}

# colores en BGR porque OpenCV usa ese orden
COLORES = {
    2: (235, 160, 40),   # azul
    3: (60, 200, 60),    # verde
    5: (40, 170, 250),   # naranja
    7: (60, 60, 230),    # rojo
}

COLOR_NIVEL = {
    "Bajo": (80, 200, 80),
    "Medio": (40, 200, 240),
    "Alto": (60, 60, 230),
}


class AnalizadorTrafico:
    def __init__(
        self,
        modelo="yolo11n.pt",
        confianza=0.35,
        posicion_linea=0.6,
        capacidad=25,
        fps=30.0,
        tamano_imagen=640,
    ):
        """
        modelo: pesos de YOLO, usamos el n porque es el mas liviano
        confianza: confianza minima para aceptar una deteccion
        posicion_linea: altura de la linea (0.6 = al 60 % del alto)
        capacidad: cuantos vehiculos en pantalla tomamos como via llena
        fps: para pasar la velocidad a por segundo
        """
        self.modelo = YOLO(modelo)
        self.confianza = confianza
        self.posicion_linea = posicion_linea
        self.capacidad = capacidad
        self.fps = fps if fps and fps > 0 else 30.0
        self.tamano_imagen = tamano_imagen
        self.reiniciar()

    def reiniciar(self):
        """Deja todo en cero para empezar un video nuevo."""
        self.conteo_clase = {c: 0 for c in CLASES_VEHICULO}  # cuantos cruzaron la linea de cada tipo
        self.conteo_direccion = {"bajando": 0, "subiendo": 0}
        self.ids_contados = set()  # para no contar dos veces el mismo
        self.ids_vistos = {c: set() for c in CLASES_VEHICULO}  # todos los IDs que aparecieron
        self.ultima_pos = {}  # centro de cada ID en el frame anterior
        self.historial_pos = defaultdict(lambda: deque(maxlen=15))  # ultimos 15 centros (medio segundo)
        self.historial_vel = defaultdict(lambda: deque(maxlen=10))  # ultimas 10 velocidades
        self.historial_indice = deque(maxlen=int(self.fps * 2))  # indice de los ultimos 2 segundos
        self.frame_idx = 0


    def procesar_frame(self, frame):
        """Procesa un frame y devuelve la imagen dibujada y las estadisticas."""
        alto, ancho = frame.shape[:2]
        y_linea = int(alto * self.posicion_linea)
        self.frame_idx += 1

        # deteccion y seguimiento. persist=True es para que no se pierdan los IDs entre frames
        resultados = self.modelo.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=list(CLASES_VEHICULO.keys()),
            conf=self.confianza,
            imgsz=self.tamano_imagen,
            verbose=False,
        )[0]

        detecciones = []
        if resultados.boxes is not None and resultados.boxes.id is not None:
            cajas = resultados.boxes.xyxy.cpu().numpy()
            ids = resultados.boxes.id.cpu().numpy().astype(int)
            clases = resultados.boxes.cls.cpu().numpy().astype(int)
            confs = resultados.boxes.conf.cpu().numpy()
            detecciones = list(zip(cajas, ids, clases, confs))

        ids_actuales = set()
        en_pantalla = {c: 0 for c in CLASES_VEHICULO}
        velocidades = []

        for (x1, y1, x2, y2), id_, clase, conf in detecciones:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ids_actuales.add(id_)
            en_pantalla[clase] += 1
            self.ids_vistos[clase].add(id_)

            # conteo: cruzo si el centro paso de un lado de la linea al otro.
            # ademas tiene que venir moviendose hacia arriba o abajo y no de lado, si no
            # se cuentan los carros de la calle de al lado en la interseccion
            self.historial_pos[id_].append((cx, cy))
            if id_ in self.ultima_pos:
                px, py = self.ultima_pos[id_]
                cruzo = (py >= y_linea) != (cy >= y_linea)
                hx, hy = self.historial_pos[id_][0]
                vertical = abs(cy - hy) > 0.3 * abs(cx - hx)
                if id_ not in self.ids_contados and cruzo and vertical:
                    self.ids_contados.add(id_)
                    self.conteo_clase[clase] += 1
                    self.conteo_direccion["bajando" if cy > py else "subiendo"] += 1

                # velocidad en pixeles por segundo, dividida por el alto para que no dependa de la resolucion
                desplazamiento = np.hypot(cx - px, cy - py)
                self.historial_vel[id_].append(desplazamiento * self.fps / alto)

            if self.historial_vel[id_]:
                velocidades.append(float(np.mean(self.historial_vel[id_])))
            self.ultima_pos[id_] = (cx, cy)

        # borrar los que ya salieron del video
        for id_ in list(self.ultima_pos):
            if id_ not in ids_actuales and self.frame_idx % 30 == 0:
                self.ultima_pos.pop(id_, None)
                self.historial_vel.pop(id_, None)
                self.historial_pos.pop(id_, None)

        # congestion
        total_en_pantalla = sum(en_pantalla.values())
        vel_media = float(np.mean(velocidades)) if velocidades else 0.0
        indice, nivel = self._estimar_congestion(total_en_pantalla, vel_media)

        estadisticas = {
            "frame": self.frame_idx,
            "tiempo_s": round(self.frame_idx / self.fps, 2),
            "en_pantalla": total_en_pantalla,
            "en_pantalla_por_clase": {CLASES_VEHICULO[c]: n for c, n in en_pantalla.items()},
            "contados_por_clase": {CLASES_VEHICULO[c]: n for c, n in self.conteo_clase.items()},
            "total_contados": sum(self.conteo_clase.values()),
            "direccion": dict(self.conteo_direccion),
            "unicos_por_clase": {CLASES_VEHICULO[c]: len(s) for c, s in self.ids_vistos.items()},
            "velocidad_media": round(vel_media, 3),
            "indice_congestion": round(indice, 3),
            "nivel_congestion": nivel,
        }

        # dibujar
        anotado = self._dibujar(frame.copy(), detecciones, y_linea, estadisticas)
        return anotado, estadisticas


    def _estimar_congestion(self, n_vehiculos, vel_media, vel_libre=0.25):
        """
        indice = 0.6 * densidad + 0.4 * lentitud (da entre 0 y 1)
        densidad = vehiculos en pantalla / capacidad
        lentitud = 1 - velocidad media / vel_libre
        Si no hay carros la lentitud es 0, y con menos de 5 carros pesa menos.
        vel_libre = 0.25 lo sacamos probando con los videos.
        """
        densidad = min(n_vehiculos / self.capacidad, 1.0)
        if n_vehiculos == 0:
            lentitud = 0.0
        else:
            lentitud = 1.0 - min(vel_media / vel_libre, 1.0)
        indice_instantaneo = 0.6 * densidad + 0.4 * lentitud * min(n_vehiculos / 5, 1.0)

        self.historial_indice.append(indice_instantaneo)
        indice = float(np.mean(self.historial_indice))  # promedio de los ultimos 2 s para que no salte

        if indice < 0.35:
            nivel = "Bajo"
        elif indice < 0.65:
            nivel = "Medio"
        else:
            nivel = "Alto"
        return indice, nivel


    def _dibujar(self, img, detecciones, y_linea, est):
        alto, ancho = img.shape[:2]
        escala = max(ancho / 1280, 0.5)
        grosor = max(int(2 * escala), 1)

        # cajas
        for (x1, y1, x2, y2), id_, clase, conf in detecciones:
            color = COLORES[clase]
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(img, p1, p2, color, grosor)
            texto = f"{CLASES_VEHICULO[clase]} #{id_} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * escala, 1)
            # si la etiqueta no cabe arriba de la caja se pone adentro
            base = p1[1] if p1[1] - th - 6 >= 0 else p1[1] + th + 6
            cv2.rectangle(img, (p1[0], base - th - 6), (p1[0] + tw + 4, base), color, -1)
            cv2.putText(img, texto, (p1[0] + 2, base - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * escala, (255, 255, 255), 1, cv2.LINE_AA)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(img, (cx, cy), max(int(3 * escala), 2), color, -1)

        # linea de conteo
        cv2.line(img, (0, y_linea), (ancho, y_linea), (0, 255, 255), grosor + 1)
        cv2.putText(img, "LINEA DE CONTEO", (10, y_linea - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 * escala, (0, 255, 255), grosor, cv2.LINE_AA)

        # panel con los conteos
        filas = [
            f"Total contados: {est['total_contados']}",
            *[f"  {k}: {v}" for k, v in est["contados_por_clase"].items()],
            f"En pantalla: {est['en_pantalla']}",
            f"Bajando: {est['direccion']['bajando']}  Subiendo: {est['direccion']['subiendo']}",
        ]
        alto_fila = int(26 * escala)
        panel_h = alto_fila * (len(filas) + 2) + 10
        panel_w = int(330 * escala)
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        y = 10 + alto_fila
        for fila in filas:
            cv2.putText(img, fila, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * escala,
                        (255, 255, 255), 1, cv2.LINE_AA)
            y += alto_fila

        # barra de congestion
        nivel, indice = est["nivel_congestion"], est["indice_congestion"]
        color = COLOR_NIVEL[nivel]
        cv2.putText(img, f"Congestion: {nivel} ({indice:.2f})", (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * escala, color, grosor, cv2.LINE_AA)
        y += int(10 * escala)
        barra_w = panel_w - 20
        cv2.rectangle(img, (20, y), (20 + barra_w, y + int(12 * escala)), (90, 90, 90), -1)
        cv2.rectangle(img, (20, y), (20 + int(barra_w * indice), y + int(12 * escala)), color, -1)
        return img
