# -*- coding: utf-8 -*-
"""Servidor Flask mínimo del Tema 13: sirve el tablero HTML y el JSON del sistema de recomendaciones.
Uso: pip install flask && python servidor_flask_tablero.py   (abre http://127.0.0.1:5013/)
Coloque en la misma carpeta 08_Tablero_Clasificacion.html y dashboard_clasificacion.json (salida del notebook G5 o del script m4)."""
import json, os
from flask import Flask, jsonify, send_from_directory, abort

AQUI = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)


@app.get("/")
def tablero():
    return send_from_directory(AQUI, "08_Tablero_Clasificacion.html")


@app.get("/api/dashboard")
def api_dashboard():
    ruta = os.path.join(AQUI, "dashboard_clasificacion.json")
    if not os.path.exists(ruta):
        abort(404, "Falta dashboard_clasificacion.json: ejecute el notebook G5 o el script m4")
    with open(ruta, encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.get("/api/recomendaciones")
def api_recomendaciones():
    ruta = os.path.join(AQUI, "dashboard_clasificacion.json")
    with open(ruta, encoding="utf-8") as f:
        d = json.load(f)
    return jsonify({"fecha_corte": d["fecha_corte"], "aviso": d["aviso"], "recomendaciones": d["recomendaciones"]})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5013, debug=False)
