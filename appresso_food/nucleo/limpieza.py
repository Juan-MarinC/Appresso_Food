"""Limpieza y normalización de datos del negocio y reporte de calidad.

Problemas que ataca (los del enunciado): ítems duplicados escritos distinto ("Papas fritas" / "papas  FRITAS"),
precios mal escritos ("$12.000", "12,000.50"), categorías inconsistentes ("bebidas " / "Bebidas") y clientes
registrados dos veces ("ana lópez" / "Ana Lopez").

Se usa en las fronteras (al crear productos y pedidos) y en GET /api/calidad.
"""

import re
import unicodedata
from typing import Dict, Iterable, List, Optional

from appresso_food.nucleo.counter import OperationCounter


def normalizar_texto(texto) -> str:
    """Quita espacios sobrantes: "  Papas   fritas " -> "Papas fritas". O(n) en el largo del texto."""
    return " ".join(str(texto).split())


def clave(texto) -> str:
    """Forma para comparar: sin tildes, mayúsculas ni signos. "Hamburguesa  Clásica!" == "hamburguesa clasica"."""
    sin_tildes = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", sin_tildes.lower()).split())


def nombre_propio(texto) -> str:
    """Nombre de persona unificado: "ana  LÓPEZ" -> "Ana López" (conserva las tildes)."""
    return " ".join(p[:1].upper() + p[1:].lower() for p in normalizar_texto(texto).split(" ") if p)


def parse_precio(valor) -> Optional[float]:
    """Precio mal escrito a número: "$12.000" -> 12000, "12.000,50" -> 12000.5, "12,000.50" -> 12000.5.

    En Colombia el punto separa miles y la coma decimales; si el texto trae los dos, decide el último que aparece.
    Devuelve None si no es un precio válido (vacío, letras, negativo o cero).
    """
    if isinstance(valor, bool):
        return None
    if isinstance(valor, (int, float)):
        numero = float(valor)
    else:
        texto = re.sub(r"[\s$]|COP", "", str(valor), flags=re.I)
        if not re.fullmatch(r"\d[\d.,]*", texto):
            return None
        if "," in texto and "." in texto:
            decimal = "," if texto.rfind(",") > texto.rfind(".") else "."
            miles = "." if decimal == "," else ","
            texto = texto.replace(miles, "").replace(decimal, ".")
        elif "," in texto:
            enteros, _, decimales = texto.rpartition(",")
            texto = texto.replace(",", "") if len(decimales) == 3 else enteros.replace(",", "") + "." + decimales
        elif texto.count(".") > 1 or re.fullmatch(r"\d{1,3}\.\d{3}", texto):
            texto = texto.replace(".", "")  # 12.000 o 1.250.000: puntos de miles
        numero = float(texto)
    return numero if numero > 0 else None


def duplicados(valores: Iterable[str], counter: Optional[OperationCounter] = None) -> Dict[str, List[str]]:
    """Agrupa los valores que son el mismo una vez normalizados. O(n): un diccionario por clave.

    Devuelve solo los grupos con más de una forma de escribirlo: {"papas fritas": ["Papas Fritas", "papas  fritas"]}.
    """
    grupos: Dict[str, List[str]] = {}
    for valor in valores:
        if counter:
            counter.comparisons += 1  # una consulta al diccionario por valor
        grupos.setdefault(clave(valor), []).append(valor)
    return {k: v for k, v in grupos.items() if len(v) > 1}


def duplicados_ingenuo(valores: List[str], counter: OperationCounter) -> int:
    """La forma ingenua de lo mismo: comparar cada valor con todos los siguientes. O(n²): n(n-1)/2 comparaciones.

    Se deja para el laboratorio, para medir cuánto mejora `duplicados` (mismo resultado, otra complejidad).
    Devuelve cuántos valores tienen al menos un repetido.
    """
    claves = [clave(v) for v in valores]
    repetidos = set()
    for i in range(len(claves)):
        for j in range(i + 1, len(claves)):
            counter.comparisons += 1
            if claves[i] == claves[j]:
                repetidos.update((i, j))
    return len(repetidos)


def reporte_calidad(productos, pedidos, ingredientes) -> dict:
    """Problemas de calidad encontrados en los datos de SQLite, con ejemplos para corregirlos."""
    problemas = []

    def agregar(modulo: str, tipo: str, detalle: str, ejemplos: List) -> None:
        if ejemplos:
            problemas.append({"modulo": modulo, "tipo": tipo, "detalle": detalle, "cantidad": len(ejemplos), "ejemplos": ejemplos[:5]})

    agregar("ítems", "duplicados", "El mismo producto escrito de varias formas",
            [formas for formas in duplicados(p.name for p in productos).values()])
    agregar("ítems", "precio inválido", "Precio en cero, negativo o vacío",
            [p.name for p in productos if not p.price or p.price <= 0])
    categorias = {}
    for p in productos:
        categorias.setdefault(clave(p.category), set()).add(p.category)
    agregar("categorías", "inconsistentes", "La misma categoría escrita distinto",
            [sorted(formas) for formas in categorias.values() if len(formas) > 1])
    agregar("clientes", "registrados dos veces", "El mismo cliente con nombres escritos distinto",
            [sorted(set(formas)) for formas in duplicados(o.client for o in pedidos).values() if len(set(formas)) > 1])
    agregar("pedidos", "sin ítems", "Pedidos sin productos", [o.order_id for o in pedidos if not o.items])
    agregar("inventario", "stock negativo", "Ingredientes con existencias por debajo de cero",
            [i.name for i in ingredientes if i.stock_qty < 0])

    revisados = {"ítems": len(productos), "categorías": len(categorias), "pedidos": len(pedidos), "ingredientes": len(ingredientes)}
    con_problema = sum(p["cantidad"] for p in problemas)
    total = sum(revisados.values()) or 1
    return {
        "revisados": revisados,
        "problemas": problemas,
        "puntaje": round(100 * max(0.0, 1 - con_problema / total), 1),  # % de registros sin problemas
    }
