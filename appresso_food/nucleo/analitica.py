"""Patrones de operación y comportamiento de empleados para el panel.

- Patrones: pedidos por hora y por canal, montos (cuartiles y atípicos por rango intercuartílico) y frecuencia por cliente.
- Empleados: facturas, descuentos y anulaciones por cajero; se marca a quien se sale de lo normal frente a los demás
  (descuento promedio por encima de Q3 + 1,5·RIC, o más de la mitad de sus facturas anuladas).
"""

from collections import Counter
from statistics import mean, quantiles
from typing import Dict, List

from appresso_food.nucleo.facturacion import vigentes as facturas_vigentes
from appresso_food.nucleo.limpieza import clave


def cuartiles(valores: List[float]) -> Dict[str, float]:
    """Q1, mediana, Q3, rango intercuartílico y los límites para valores atípicos (regla de Tukey)."""
    if len(valores) < 4:
        return {}
    q1, q2, q3 = quantiles(valores, n=4)
    ric = q3 - q1
    return {"q1": q1, "mediana": q2, "q3": q3, "ric": ric, "limite_bajo": q1 - 1.5 * ric, "limite_alto": q3 + 1.5 * ric}


def patrones(pedidos, facturas) -> dict:
    vigentes = facturas_vigentes(facturas)
    montos = [f["total"] for f in vigentes]
    q = cuartiles(montos)
    atipicos = [f for f in vigentes if q and not q["limite_bajo"] <= f["total"] <= q["limite_alto"]]
    por_cliente = Counter(clave(p.client) for p in pedidos)  # "Ana López" y "ana lopez" cuentan como uno
    nombre_de = {}
    for p in pedidos:
        nombre_de.setdefault(clave(p.client), p.client)  # se muestra la primera forma en que se escribió
    return {
        "por_hora": sorted(Counter(p.created_at[11:13] for p in pedidos if len(p.created_at) >= 13).items()),
        "por_canal": dict(Counter(p.channel for p in pedidos)),
        "montos": q,
        "montos_atipicos": [{"factura": f["number"], "total": f["total"]} for f in atipicos[:10]],
        "clientes_frecuentes": [(nombre_de[c], n) for c, n in por_cliente.most_common(5)],
        "ventas": sum(montos),
        "ticket_promedio": mean(montos) if montos else 0,
        "facturas": len(vigentes),
        "anuladas": len(facturas) - len(vigentes) - sum(1 for f in facturas if f["kind"] == "ANULACION"),
    }


def empleados(facturas) -> List[dict]:
    """Una fila por quien emite facturas (cajero), con alerta si su comportamiento se sale del grupo."""
    filas: Dict[str, dict] = {}
    for f in facturas:
        quien = f["issued_by"] or "Kiosko / cliente"
        fila = filas.setdefault(quien, {"empleado": quien, "facturas": 0, "con_descuento": 0, "descuentos": [], "anulaciones": 0})
        if f["kind"] == "FACTURA":
            fila["facturas"] += 1
            fila["descuentos"].append(f["discount_pct"])
            fila["con_descuento"] += f["discount_pct"] > 0
        else:
            fila["anulaciones"] += 1
    for fila in filas.values():
        fila["descuento_promedio"] = mean(fila.pop("descuentos")) if fila["facturas"] else 0
    q = cuartiles([f["descuento_promedio"] for f in filas.values()])
    for fila in filas.values():
        alertas = []
        if q and fila["descuento_promedio"] > max(q["limite_alto"], 0) and fila["descuento_promedio"] > 0:
            alertas.append("descuentos muy por encima de los demás")
        if fila["facturas"] and fila["anulaciones"] / fila["facturas"] > 0.5:
            alertas.append("anula más de la mitad de sus facturas")
        fila["alertas"] = alertas
    return sorted(filas.values(), key=lambda f: (-len(f["alertas"]), -f["facturas"]))
