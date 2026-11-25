"""Algoritmos base sobre pedidos reales, instrumentados con OperationCounter
para poder justificar su complejidad (Big O) frente al profesor.
"""

from typing import List, Optional

from appresso_food.nucleo.counter import OperationCounter
from appresso_food.nucleo.models import Order


def find_order_linear(orders: List[Order], target_id: int, counter: OperationCounter) -> Optional[Order]:
    """Búsqueda lineal de un pedido por id. O(n) en el peor caso, O(1) en el mejor.
    """
    for order in orders:
        counter.comparisons += 1
        if order.order_id == target_id:
            return order
    return None


def total_sales(orders: List[Order], counter: OperationCounter) -> float:
    """Suma de ventas: el total de cada pedido, una suma por pedido. Exactamente n sumas: O(n)."""
    total = 0.0
    for order in orders:
        counter.additions += 1
        total += order.total
    return total


def recursive_total_sales(orders: List[Order], counter: OperationCounter) -> float:
    """La misma suma de ventas por divide y vencerás: parte la lista en mitades hasta llegar a un pedido.

    Hace 2n - 1 llamadas: n que llegan a un pedido y devuelven su total, y n - 1 que parten la lista y suman
    las dos mitades (con n = 10: 10 + 9 = 19). Trabajo O(n); lo logarítmico es solo la profundidad, O(log n).
    """
    def analyze_range(start: int, end: int, depth: int = 1) -> int:
        counter.calls += 1
        counter.max_depth = max(counter.max_depth, depth)
        if start >= end:
            return 0
        if end - start == 1:
            return orders[start].total

        midpoint = start + (end - start) // 2
        left_total = analyze_range(start, midpoint, depth + 1)
        right_total = analyze_range(midpoint, end, depth + 1)
        counter.additions += 1
        return left_total + right_total

    return analyze_range(0, len(orders))


# ---------------------------------------------------------------------------------------------
# Algoritmos del final: conteos, ordenamientos y ranking, en versión ingenua y optimizada.
# Todos reciben un OperationCounter para comparar operaciones además de tiempos.
# ---------------------------------------------------------------------------------------------

def count_orders_per_client_naive(orders: List[Order], counter: OperationCounter) -> dict:
    """Conteo de pedidos por cliente comparando cada pedido con todos los demás. O(n²): n·n comparaciones."""
    conteo = {}
    for order in orders:
        total = 0
        for other in orders:
            counter.comparisons += 1
            if other.client == order.client:
                total += 1
        conteo[order.client] = total
    return conteo


def count_orders_per_client(orders: List[Order], counter: OperationCounter) -> dict:
    """Lo mismo con un diccionario: una pasada. O(n). Es la versión optimizada de la anterior."""
    conteo = {}
    for order in orders:
        counter.comparisons += 1  # una consulta al diccionario
        conteo[order.client] = conteo.get(order.client, 0) + 1
    return conteo


def insertion_sort_by_total(orders: List[Order], counter: OperationCounter) -> List[Order]:
    """Ordenamiento simple (inserción) por total del pedido. O(n²) en el caso promedio y peor; O(n) si ya viene ordenado."""
    resultado = list(orders)
    for i in range(1, len(resultado)):
        actual = resultado[i]
        j = i - 1
        while j >= 0:
            counter.comparisons += 1
            if resultado[j].total <= actual.total:
                break
            resultado[j + 1] = resultado[j]
            j -= 1
        resultado[j + 1] = actual
    return resultado


def merge_sort_by_date(orders: List[Order], counter: OperationCounter) -> List[Order]:
    """Ordenamiento por fecha con merge sort (divide y vencerás). O(n log n) siempre, y estable."""
    if len(orders) <= 1:
        return list(orders)
    mitad = len(orders) // 2
    izquierda = merge_sort_by_date(orders[:mitad], counter)
    derecha = merge_sort_by_date(orders[mitad:], counter)
    mezcla, i, j = [], 0, 0
    while i < len(izquierda) and j < len(derecha):
        counter.comparisons += 1
        if izquierda[i].created_at <= derecha[j].created_at:
            mezcla.append(izquierda[i])
            i += 1
        else:
            mezcla.append(derecha[j])
            j += 1
    return mezcla + izquierda[i:] + derecha[j:]


def top_items(orders: List[Order], counter: OperationCounter, k: int = 5) -> List[tuple]:
    """Ranking de ítems más vendidos: acumula unidades por producto (O(n)) y ordena los m productos (O(m log m)).

    Como el menú tiene pocos productos (m << n), en la práctica crece como O(n).
    """
    unidades = {}
    for order in orders:
        for item in order.items:
            counter.additions += 1
            unidades[item.product_name] = unidades.get(item.product_name, 0) + item.quantity
    return sorted(unidades.items(), key=lambda par: par[1], reverse=True)[:k]
