"""Facturación, pagos con comprobante e integridad por hash encadenado.

Cada factura (o anulación) y cada pago (o comprobante) es una fila que solo se agrega. Su hash es
SHA-256 de sus datos *junto con el hash de la fila anterior*:

    hash_n = SHA-256( datos_n + hash_(n-1) )        hash_0 parte de GENESIS (64 ceros)

Así, cambiar una factura vieja (por ejemplo el total) cambia su hash y rompe todos los eslabones siguientes;
borrar o insertar una fila deja un prev_hash que ya no coincide. `verificar_cadena` dice en qué fila se rompió.

Impuesto: los restaurantes en Colombia cobran el impuesto nacional al consumo (8 %), incluido en el precio del menú.
"""

import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from appresso_food.nucleo import storage

TASA_IMPOCONSUMO = 0.08
DESCUENTO_MAXIMO = 30  # % que puede dar un cajero; por encima se rechaza
GENESIS = "0" * 64
TIPOS_COMPROBANTE = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".pdf": "application/pdf"}
MAX_COMPROBANTE = 2_000_000  # bytes
_lock = threading.RLock()  # leer consecutivo y último hash e insertar debe ser atómico (o se repetirían número o prev_hash)


class FacturaError(ValueError):
    pass


def _canon(valor):
    # SQLite devuelve 16000.0 aunque se guarde 16000: los números se hashean siempre como float con 2 decimales.
    if isinstance(valor, (int, float)) and not isinstance(valor, bool):
        return round(float(valor), 2) + 0.0  # + 0.0 convierte -0.0 en 0.0, como lo devuelve SQLite
    return valor


def calcular_hash(fila: dict) -> str:
    datos = {k: _canon(v) for k, v in fila.items() if k not in ("id", "hash")}  # incluye prev_hash: eso encadena
    return hashlib.sha256(json.dumps(datos, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _encadenar(tabla: str, fila: dict) -> dict:
    with _lock:
        fila["prev_hash"] = storage.last_hash(tabla) or GENESIS
        fila["hash"] = calcular_hash(fila)
        storage.insert_chained(tabla, fila)
    return fila


def _ahora() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --- Facturas ----------------------------------------------------------------------------------

def calcular(items, descuento_pct: float = 0) -> dict:
    """Subtotal, descuento, impuesto (incluido) y total en pesos enteros."""
    if not 0 <= descuento_pct <= DESCUENTO_MAXIMO:
        raise FacturaError(f"El descuento debe estar entre 0 y {DESCUENTO_MAXIMO} %.")
    subtotal = round(sum(i.quantity * i.unit_price for i in items))
    descuento = round(subtotal * descuento_pct / 100)
    total = subtotal - descuento
    impuesto = round(total - total / (1 + TASA_IMPOCONSUMO))
    return {"subtotal": subtotal, "discount_pct": descuento_pct, "discount": descuento, "tax": impuesto, "total": total}


def emitir_factura(order_id: int, items, descuento_pct: float = 0, emitida_por: Optional[str] = None) -> dict:
    montos = calcular(items, descuento_pct)
    with _lock:
        numero = f"FV-{storage.count_invoices('FACTURA') + 1:06d}"  # consecutivo, sin huecos
        return _encadenar("invoices", {"number": numero, "kind": "FACTURA", "order_id": order_id, "refers_to": None, **montos,
                                       "reason": None, "issued_by": emitida_por, "created_at": _ahora()})


def anuladas(facturas: List[dict]) -> set:
    """Números de las facturas que tienen una nota crédito (anulación) en la cadena."""
    return {f["refers_to"] for f in facturas if f["kind"] == "ANULACION"}


def con_estado(facturas: List[dict]) -> List[dict]:
    """Cada fila con su marca "anulada" (para listarlas)."""
    nulas = anuladas(facturas)
    return [{**f, "anulada": f["number"] in nulas} for f in facturas]


def vigentes(facturas: List[dict]) -> List[dict]:
    """Facturas emitidas que no fueron anuladas."""
    nulas = anuladas(facturas)
    return [f for f in facturas if f["kind"] == "FACTURA" and f["number"] not in nulas]


def factura_de(order_id: int) -> Optional[dict]:
    """La factura vigente (no anulada) de un pedido."""
    del_pedido = [f for f in vigentes(storage.get_invoices()) if f["order_id"] == order_id]
    return del_pedido[-1] if del_pedido else None


def anular_factura(numero: str, motivo: str, anulada_por: Optional[str] = None) -> dict:
    """La factura no se borra ni se edita: se emite una nota crédito (NC) que la anula, también encadenada."""
    facturas = storage.get_invoices()
    factura = next((f for f in facturas if f["number"] == numero and f["kind"] == "FACTURA"), None)
    if factura is None:
        raise FacturaError(f"No existe la factura {numero}.")
    if numero in anuladas(facturas):
        raise FacturaError(f"La factura {numero} ya estaba anulada.")
    if not motivo.strip():
        raise FacturaError("Toda anulación necesita un motivo.")
    negativos = {k: -factura[k] for k in ("subtotal", "discount", "tax", "total")}
    with _lock:
        nc = f"NC-{storage.count_invoices('ANULACION') + 1:06d}"
        return _encadenar("invoices", {"number": nc, "kind": "ANULACION", "order_id": factura["order_id"], "refers_to": numero,
                                       "discount_pct": factura["discount_pct"], **negativos, "reason": motivo.strip()[:200],
                                       "issued_by": anulada_por, "created_at": _ahora()})


# --- Pagos y comprobantes ----------------------------------------------------------------------

def registrar_pago(order_id: int, id_txn: str, metodo: Optional[str], monto: float) -> dict:
    factura = factura_de(order_id)
    return _encadenar("payments", {"kind": "PAGO", "order_id": order_id, "invoice_number": factura and factura["number"],
                                   "method": metodo, "amount": monto, "id_txn": id_txn, "receipt_name": None,
                                   "receipt_sha256": None, "created_at": _ahora()})


def carpeta_comprobantes() -> Path:
    return Path(storage.DB_PATH).parent / "comprobantes"  # en Docker: /data/comprobantes (volumen)


def registrar_comprobante(order_id: int, nombre: str, contenido: bytes) -> dict:
    """Guarda el archivo con su huella SHA-256 como nombre y la deja en la cadena: si alguien cambia el archivo
    después, la huella ya no coincide (`verificar_comprobantes`)."""
    extension = Path(nombre).suffix.lower()
    if extension not in TIPOS_COMPROBANTE:
        raise FacturaError("El comprobante debe ser una imagen PNG o JPG, o un PDF.")
    if not contenido or len(contenido) > MAX_COMPROBANTE:
        raise FacturaError(f"El comprobante debe pesar entre 1 byte y {MAX_COMPROBANTE // 1_000_000} MB.")
    huella = hashlib.sha256(contenido).hexdigest()
    carpeta = carpeta_comprobantes()
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / f"{huella}{extension}").write_bytes(contenido)
    factura = factura_de(order_id)
    return _encadenar("payments", {"kind": "COMPROBANTE", "order_id": order_id, "invoice_number": factura and factura["number"],
                                   "method": None, "amount": None, "id_txn": None, "receipt_name": Path(nombre).name[:120],
                                   "receipt_sha256": huella, "created_at": _ahora()})


# --- Verificación ------------------------------------------------------------------------------

def verificar_cadena(filas: List[dict]) -> dict:
    """Recorre la cadena desde el inicio y dice si está íntegra o en qué fila se rompió y por qué."""
    anterior = GENESIS
    for fila in filas:
        if fila["prev_hash"] != anterior:
            return {"ok": False, "registros": len(filas), "roto_en": fila["id"],
                    "motivo": "Falta o sobra una fila antes de esta: su hash anterior no coincide (¿la borraron o insertaron?)."}
        if calcular_hash(fila) != fila["hash"]:
            return {"ok": False, "registros": len(filas), "roto_en": fila["id"],
                    "motivo": "Los datos de esta fila cambiaron después de guardarla: su hash ya no coincide."}
        anterior = fila["hash"]
    return {"ok": True, "registros": len(filas), "roto_en": None, "motivo": None}


def verificar_comprobantes() -> List[dict]:
    """Comprobantes cuyo archivo falta o fue alterado (su SHA-256 ya no es el registrado)."""
    problemas = []
    for fila in storage.get_payments():
        if fila["kind"] != "COMPROBANTE":
            continue
        archivos = list(carpeta_comprobantes().glob(f"{fila['receipt_sha256']}.*"))
        if not archivos:
            problemas.append({"pedido": fila["order_id"], "archivo": fila["receipt_name"], "problema": "El archivo no está."})
        elif hashlib.sha256(archivos[0].read_bytes()).hexdigest() != fila["receipt_sha256"]:
            problemas.append({"pedido": fila["order_id"], "archivo": fila["receipt_name"], "problema": "El archivo fue modificado."})
    return problemas


def integridad() -> dict:
    return {
        "facturas": verificar_cadena(storage.get_invoices()),
        "pagos": verificar_cadena(storage.get_payments()),
        "comprobantes_con_problema": verificar_comprobantes(),
    }
