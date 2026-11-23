"""Datos para los modelos de IA de /ia (se entrenan en el navegador con TensorFlow.js y Simple Statistics).

- /api/ia/operaciones: cada transacción de la pasarela con lo que pide el enunciado (usuario, fecha, monto, método
  de pago, firma y frecuencia) y su etiqueta: normal o fraudulenta según lo que detectó el antifraude.
- /api/ia/demanda: unidades vendidas por día, para predecir la demanda.

Sin datos personales: el usuario va como un seudónimo (los primeros caracteres del SHA-256 de su correo).
"""

import hashlib
import random
from collections import Counter
from datetime import datetime
from typing import Optional

from fastapi import APIRouter

from appresso_food.antifraude import fraud_routes
from appresso_food.nucleo import storage

router = APIRouter()

MAX_OPERACIONES = 10_000
# Errores de formato (JSON roto, campos vacíos…) no son operaciones: no entran al entrenamiento.
FRAUDE_RECHAZADA = {"HASH_MISMATCH", "DUPLICATE_TRANSACTION"}  # firma alterada o pago repetido


def _fecha(valor) -> Optional[str]:
    if isinstance(valor, datetime):
        return valor.isoformat(timespec="seconds")
    return str(valor)[:19] if valor else None


def _seudonimo(email: Optional[str]) -> str:
    return hashlib.sha256((email or "").lower().encode()).hexdigest()[:10] if email else "anonimo"


def _firma(txn: dict) -> int:
    """1 si el cliente firmó y la firma coincidió, -1 si la firma no coincidía, 0 si no envió firma."""
    if txn.get("motivo") == "HASH_MISMATCH":
        return -1
    return 1 if txn.get("hash_origen") == "CLIENTE_VERIFICADO" else 0


@router.get("/api/ia/operaciones")
def operations_dataset():
    try:
        txns = fraud_routes.get_service().repo.list_transactions(None, None, MAX_OPERACIONES)
    except Exception:  # MongoDB caído: la página lo explica en vez de romperse
        return {"ok": False, "mensaje": "La pasarela no está disponible (MongoDB).", "operaciones": []}
    operaciones = []
    for t in txns:
        fraudulenta = t.get("estado") == "SUSPICIOUS" or t.get("motivo") in FRAUDE_RECHAZADA
        if t.get("estado") == "REJECTED" and not fraudulenta:
            continue
        fecha = _fecha(t.get("fecha_ref") or t.get("fecha_txn") or t.get("fecha_creacion"))
        if not fecha or not isinstance(t.get("valor"), (int, float)):
            continue
        operaciones.append({
            "id": t.get("id_txn"), "usuario": _seudonimo(t.get("usuario_email")), "fecha": fecha,
            "monto": float(t["valor"]), "metodo": t.get("metodo_pago") or "Otro", "firma": _firma(t),
            "frecuencia": int(t.get("cantidad_ventana") or 1), "estado": t.get("estado"),
            "fraudulenta": int(fraudulenta),
        })
    operaciones.sort(key=lambda o: o["fecha"])
    por_usuario = Counter(o["usuario"] for o in operaciones)
    for o in operaciones:
        o["operaciones_usuario"] = por_usuario[o["usuario"]]
    return {"ok": True, "operaciones": operaciones, "fraudulentas": sum(o["fraudulenta"] for o in operaciones)}


@router.get("/api/ia/demanda")
def demand_series():
    """Unidades por día. Con menos de 14 días con ventas, también va un histórico simulado de 90 días (marcado así)."""
    reales = storage.daily_totals()
    respuesta = {"reales": reales, "simulado": None}
    if sum(1 for unidades in reales if unidades) < 14:  # daily_totals rellena con 0 los días sin pedidos
        # Simulado y reproducible: base de 40 unidades, crece ~0,3 por día, +25 % los viernes y sábados, y ruido.
        rng = random.Random(11)
        respuesta["simulado"] = [
            round((40 + 0.3 * dia) * (1.25 if dia % 7 in (4, 5) else 1) + rng.gauss(0, 4)) for dia in range(90)
        ]
    return respuesta
