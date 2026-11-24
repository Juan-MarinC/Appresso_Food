"""Módulos operativos del final: facturas con impuesto y descuento, anulaciones, pagos y comprobantes con hash
encadenado, integridad, CRUD de usuarios y categorías, y limpieza/calidad de datos."""

import os
import re
import sqlite3
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from appresso_food import paginas
from appresso_food.api import negocio_api, productos_api
from appresso_food.nucleo import facturacion, runtime, storage
from appresso_food.nucleo.limpieza import clave, duplicados, duplicados_ingenuo, nombre_propio, parse_precio
from appresso_food.nucleo.counter import OperationCounter
from appresso_food.nucleo.algorithms import (
    count_orders_per_client, count_orders_per_client_naive, insertion_sort_by_total, merge_sort_by_date, top_items,
)
from appresso_food.nucleo.models import CHANNEL_LOCAL, OrderItem
from test_gestos import _ConBaseTemporal


class _ConApi(_ConBaseTemporal):
    def setUp(self):
        super().setUp()
        paginas._pedidos_por_token.clear()
        app = FastAPI()
        app.include_router(paginas.router)
        app.include_router(negocio_api.router)
        app.include_router(productos_api.router)
        self.api = TestClient(app)
        self.producto = next(p for p in storage.get_products() if p.name == "Hamburguesa Clásica")  # $16.000
        self.addCleanup(os.environ.pop, "APPRESSO_ENTRENAMIENTO", None)

    def _caja(self) -> str:
        """Token de la caja del local (como si se abriera la página en el equipo del servidor)."""
        os.environ["APPRESSO_ENTRENAMIENTO"] = "1"
        return re.search(r'name="token_equipo" value="([^"]+)"', self.api.get("/").text).group(1)

    def _pedir(self, **cambios):
        datos = {"client": "Ana", "channel": "local", "product_id": [str(self.producto.id)], "quantity": ["2"]}
        datos.update(cambios)
        return self.api.post("/pedidos", data=datos)


class FacturasTest(_ConApi):
    def test_el_pedido_genera_factura_con_descuento_e_impuesto_incluido(self):
        html = self._pedir(descuento="10", token_equipo=self._caja()).text
        factura = storage.get_invoices()[-1]
        self.assertEqual((factura["number"], factura["subtotal"], factura["discount"], factura["total"]), ("FV-000001", 32000, 3200, 28800))
        self.assertEqual(factura["tax"], round(28800 - 28800 / 1.08))  # impoconsumo 8 % incluido en el precio
        self.assertIn("FV-000001", html)
        self.assertIn("valor=28800", html)  # se cobra el total con descuento

    def test_consecutivo_y_descuento_fuera_de_rango(self):
        self._pedir()
        self._pedir()
        self.assertEqual([f["number"] for f in storage.get_invoices()], ["FV-000001", "FV-000002"])
        self.assertEqual(self._pedir(descuento="80", token_equipo=self._caja()).status_code, 400)
        self.assertEqual(len(storage.get_invoices()), 2)  # el descuento abusivo no crea pedido ni factura

    def test_deshacer_el_pedido_anula_la_factura_sin_borrarla(self):
        self._pedir()
        runtime.undo_last()
        tipos = [(f["number"], f["kind"], f["refers_to"]) for f in storage.get_invoices()]
        self.assertEqual(tipos, [("FV-000001", "FACTURA", None), ("NC-000001", "ANULACION", "FV-000001")])
        self.assertTrue(facturacion.integridad()["facturas"]["ok"])

    def test_solo_la_caja_del_local_puede_dar_descuentos(self):
        os.environ.pop("APPRESSO_ENTRENAMIENTO", None)
        self.assertNotIn('name="descuento"', self.api.get("/").text)  # por la URL pública ni aparece el campo
        r = self._pedir(descuento="30", token_equipo="adivinado")
        self.assertEqual(r.status_code, 400)
        self.assertIn("caja del local", r.text)
        self.assertEqual(storage.get_invoices(), [])  # no se creó pedido ni factura

    def test_anular_por_api_exige_el_token_del_equipo(self):
        self._pedir()
        r = self.api.post("/api/facturas/FV-000001/anular", json={"motivo": "prueba", "token": "adivinado"})
        self.assertEqual((r.status_code, r.json()["motivo"]), (403, "SOLO_EQUIPO"))


class IntegridadTest(_ConApi):
    def test_cambiar_una_factura_rompe_la_cadena_en_esa_fila(self):
        for _ in range(3):
            self._pedir()
        self.assertTrue(self.api.get("/api/integridad").json()["ok"])
        conn = sqlite3.connect(storage.DB_PATH)
        conn.execute("UPDATE invoices SET total = 1 WHERE number = 'FV-000002'")
        conn.commit()
        conn.close()
        estado = self.api.get("/api/integridad").json()
        self.assertFalse(estado["ok"])
        self.assertEqual(estado["facturas"]["roto_en"], 2)

    def test_borrar_una_fila_tambien_se_detecta(self):
        for _ in range(3):
            self._pedir()
        conn = sqlite3.connect(storage.DB_PATH)
        conn.execute("DELETE FROM invoices WHERE number = 'FV-000002'")
        conn.commit()
        conn.close()
        self.assertIn("hash anterior", facturacion.integridad()["facturas"]["motivo"])

    def test_comprobante_con_clave_y_alteracion_detectada(self):
        self._pedir()
        order = storage.get_all_orders()[-1]
        archivo = {"archivo": ("pago.png", b"\x89PNG comprobante", "image/png")}
        self.assertEqual(self.api.post(f"/api/pedidos/{order.order_id}/comprobante", files=archivo, data={"clave": order.pay_key}).status_code, 409)  # sin pagar
        storage.mark_order_paid(order.order_id, "TX-1")
        facturacion.registrar_pago(order.order_id, "TX-1", "Tarjeta", 32000)
        self.assertEqual(self.api.post(f"/api/pedidos/{order.order_id}/comprobante", files=archivo, data={"clave": "otra"}).status_code, 403)
        r = self.api.post(f"/api/pedidos/{order.order_id}/comprobante", files=archivo, data={"clave": order.pay_key})
        self.assertEqual(r.status_code, 200)
        self.assertEqual([p["kind"] for p in storage.get_payments()], ["PAGO", "COMPROBANTE"])
        (facturacion.carpeta_comprobantes() / f"{r.json()['sha256']}.png").write_bytes(b"otro archivo")
        self.assertEqual(facturacion.verificar_comprobantes()[0]["problema"], "El archivo fue modificado.")
        malo = {"archivo": ("virus.exe", b"MZ", "application/octet-stream")}
        self.assertEqual(self.api.post(f"/api/pedidos/{order.order_id}/comprobante", files=malo, data={"clave": order.pay_key}).status_code, 422)


class UsuariosYCategoriasTest(_ConApi):
    def test_crud_de_usuarios_con_roles(self):
        r = self.api.post("/api/usuarios", json={"nombre": "  ana  LÓPEZ ", "email": "ANA@Correo.com", "rol": "cajero"})
        self.assertEqual(r.status_code, 201)
        usuario = r.json()["usuario"]
        self.assertEqual(usuario["nombre"], "Ana López")  # limpio
        self.assertEqual(usuario["email"], "an***@correo.com")  # por la URL pública el correo va enmascarado
        self.assertEqual(storage.get_user(usuario["id"])[2], "ana@correo.com")  # guardado completo y en minúsculas
        self.assertNotIn("laura.admin@", str(self.api.get("/api/usuarios").json()))
        self.assertEqual(self.api.post("/api/usuarios", json={"nombre": "Otra", "email": "ana@correo.com", "rol": "cocina"}).status_code, 409)
        self.assertEqual(self.api.post("/api/usuarios", json={"nombre": "X", "email": "malo", "rol": "jefe"}).status_code, 422)
        self.assertEqual(self.api.patch(f"/api/usuarios/{usuario['id']}", json={"rol": "cocina"}).json()["usuario"]["rol"], "cocina")
        self.assertEqual(len(self.api.get("/api/usuarios?rol=cocina").json()), 2)
        self.assertEqual(self.api.delete(f"/api/usuarios/{usuario['id']}").status_code, 200)
        sembrado = self.api.get("/api/usuarios").json()[0]
        self.assertEqual(self.api.delete(f"/api/usuarios/{sembrado['id']}").status_code, 403)

    def test_crud_de_categorias_y_productos_unificados(self):
        r = self.api.post("/api/categorias", json={"nombre": "Postres"})
        self.assertEqual(r.status_code, 201)
        self.assertEqual(self.api.post("/api/categorias", json={"nombre": " postrés "}).status_code, 409)
        bebidas = next(c for c in self.api.get("/api/categorias").json() if c["nombre"] == "Bebidas")
        self.assertEqual(self.api.delete(f"/api/categorias/{bebidas['id']}").status_code, 403)  # es del menú
        producto = self.api.post("/api/productos", json={"nombre": "Limonada", "precio": 6000, "categoria": "  bebidas "}).json()
        self.assertEqual(producto["producto"]["categoria"], "Bebidas")  # se usó la existente
        self.assertEqual(self.api.post("/api/productos", json={"nombre": "limonáda", "precio": 1}).status_code, 409)  # duplicado
        r = self.api.put(f"/api/categorias/{bebidas['id']}", json={"nombre": "Lo que sea"})
        self.assertEqual((r.status_code, r.json()["motivo"]), (403, "CATEGORIA_PROTEGIDA"))  # la del menú no se renombra
        postres = self.api.get("/api/categorias").json()
        postres = next(c for c in postres if c["nombre"] == "Postres")
        propio = self.api.post("/api/productos", json={"nombre": "Flan", "precio": 7000, "categoria": "postres"}).json()
        self.api.put(f"/api/categorias/{postres['id']}", json={"nombre": "Postres fríos"})
        self.assertEqual(storage.get_product(propio["producto"]["id"]).category, "Postres fríos")  # el producto la sigue
        self.assertEqual(self.api.delete(f"/api/categorias/{postres['id']}").status_code, 409)  # la usa el flan


class LimpiezaTest(unittest.TestCase):
    def test_normalizacion(self):
        self.assertEqual(clave("  Hamburguesa  Clásica!"), "hamburguesa clasica")
        self.assertEqual(nombre_propio("ana  LÓPEZ"), "Ana López")
        for texto, numero in (("$12.000", 12000), ("12.000,50", 12000.5), ("12,000.50", 12000.5), ("12,5", 12.5),
                              ("1.250.000", 1250000), (16000, 16000)):
            self.assertEqual(parse_precio(texto), numero, texto)
        for malo in ("abc", "", "-5", 0, None, True):
            self.assertIsNone(parse_precio(malo), malo)

    def test_duplicados_rapido_e_ingenuo_coinciden(self):
        valores = ["Papas Fritas", "papas  fritas", "Gaseosa", "PAPÁS FRITAS", "Malteada"]
        self.assertEqual(duplicados(valores), {"papas fritas": ["Papas Fritas", "papas  fritas", "PAPÁS FRITAS"]})
        contador = OperationCounter()
        self.assertEqual(duplicados_ingenuo(valores, contador), 3)
        self.assertEqual(contador.comparisons, 5 * 4 // 2)  # n(n-1)/2


class CalidadTest(_ConApi):
    def test_el_reporte_encuentra_clientes_repetidos_y_categorias_inconsistentes(self):
        for nombre in ("Ana López", "ana lopez"):
            runtime.place_order(nombre, CHANNEL_LOCAL, None, [OrderItem(self.producto.id, self.producto.name, 1, self.producto.price)], None)
        conn = sqlite3.connect(storage.DB_PATH)
        conn.execute("UPDATE products SET category = 'bebidas' WHERE name = 'Gaseosa'")
        conn.commit()
        conn.close()
        reporte = self.api.get("/api/calidad").json()
        tipos = {p["tipo"] for p in reporte["problemas"]}
        self.assertIn("registrados dos veces", tipos)
        self.assertIn("inconsistentes", tipos)
        self.assertLess(reporte["puntaje"], 100)

    def test_el_pedido_limpia_el_nombre_del_cliente(self):
        self._pedir(client="  ana   LÓPEZ ")
        self.assertEqual(storage.get_all_orders()[-1].client, "Ana López")


class AlgoritmosFinalTest(_ConApi):
    def _pedidos(self, n):
        return paginas._pedidos_desordenados(n)

    def test_ordenamientos_correctos(self):
        pedidos = self._pedidos(300)
        por_total = insertion_sort_by_total(pedidos, OperationCounter())
        self.assertEqual([o.total for o in por_total], sorted(o.total for o in pedidos))
        por_fecha = merge_sort_by_date(pedidos, OperationCounter())
        self.assertEqual(por_fecha, sorted(pedidos, key=lambda o: o.created_at))  # mismo orden y estable, como sorted()

    def test_ingenuo_y_optimizado_dan_lo_mismo(self):
        pedidos = self._pedidos(400)
        self.assertEqual(count_orders_per_client_naive(pedidos, OperationCounter()), count_orders_per_client(pedidos, OperationCounter()))
        self.assertEqual(top_items(pedidos, OperationCounter(), k=3)[0][0], top_items(pedidos, OperationCounter(), k=10)[0][0])

    def test_las_operaciones_crecen_como_dice_su_complejidad(self):
        def ops(funcion, n):
            contador = OperationCounter()
            funcion(self._pedidos(n), contador)
            return contador.comparisons
        self.assertEqual(ops(count_orders_per_client_naive, 500) / ops(count_orders_per_client_naive, 50), 100)  # O(n²)
        self.assertEqual(ops(count_orders_per_client, 500) / ops(count_orders_per_client, 50), 10)  # O(n)
        crecimiento = ops(merge_sort_by_date, 1000) / ops(merge_sort_by_date, 100)
        self.assertTrue(10 < crecimiento < 20, crecimiento)  # O(n log n): algo más que 10

    def test_la_pagina_y_el_csv_de_la_comparacion(self):
        html = self.api.post("/laboratorio", data={"size": "25000"}).text
        self.assertIn("Comparación de algoritmos", html)
        self.assertIn("≈ estimado", html)  # 10.000 y 25.000 no se corren en O(n²)
        csv = self.api.get("/laboratorio/comparacion.csv?size=1000").text.splitlines()
        self.assertEqual(csv[3].split(",")[:3], ["1000", "no", "1000000"])  # 1000² comparaciones medidas
        self.assertEqual(self.api.get("/laboratorio/comparacion.csv?size=7").status_code, 400)


class PaginasTest(_ConApi):
    def test_volver_al_pedido_con_su_clave(self):
        self._pedir()
        order = storage.get_all_orders()[-1]
        self.assertEqual(self.api.get(f"/pedidos/{order.order_id}").status_code, 404)  # sin clave no se ve ni se confirma
        html = self.api.get(f"/pedidos/{order.order_id}?clave={order.pay_key}").text
        self.assertIn("FV-000001", html)

    def test_panel_facturas_y_administracion(self):
        self._pedir(descuento="10", served_by="Carlos", token_equipo=self._caja())
        panel = self.api.get("/panel").text
        for texto in ("Ventas facturadas", "Integridad", "Calidad de datos", "Empleados", "Benchmark", "Carlos"):
            self.assertIn(texto, panel)
        os.environ.pop("APPRESSO_ENTRENAMIENTO", None)  # de nuevo como alguien por la URL pública
        facturas = self.api.get("/facturas").text
        self.assertIn("FV-000001", facturas)
        self.assertIn("cadena íntegra", facturas)
        self.assertIn("Solo lectura", facturas)  # TestClient no es el equipo del local: no puede anular
        self.assertIn("Crear usuario", self.api.get("/administracion").text)


class IaTest(_ConApi):
    def setUp(self):
        super().setUp()
        from appresso_food.api import ia_api
        app = FastAPI()
        app.include_router(ia_api.router)
        app.include_router(paginas.router)
        self.ia = TestClient(app)

    def _con_transacciones(self, transacciones):
        from unittest import mock
        from types import SimpleNamespace
        from appresso_food.antifraude import fraud_routes
        repo = SimpleNamespace(list_transactions=lambda *args: transacciones)
        return mock.patch.object(fraud_routes, "get_service", return_value=SimpleNamespace(repo=repo))

    def test_dataset_etiquetado_y_sin_datos_personales(self):
        base = {"fecha_ref": "2026-11-20T10:00:00", "valor": 25000, "metodo_pago": "Tarjeta", "usuario_email": "ana@correo.com"}
        transacciones = [
            {**base, "id_txn": "1", "estado": "VALID", "hash_origen": "CLIENTE_VERIFICADO", "cantidad_ventana": 1},
            {**base, "id_txn": "2", "estado": "SUSPICIOUS", "motivo": "POSSIBLE_FRAUD", "cantidad_ventana": 4},
            {**base, "id_txn": "3", "estado": "REJECTED", "motivo": "HASH_MISMATCH"},
            {**base, "id_txn": "4", "estado": "REJECTED", "motivo": "MALFORMED_JSON"},  # error de formato: fuera
        ]
        with self._con_transacciones(transacciones):
            datos = self.ia.get("/api/ia/operaciones").json()
        ops = {o["id"]: o for o in datos["operaciones"]}
        self.assertEqual(sorted(ops), ["1", "2", "3"])
        self.assertEqual([ops[i]["fraudulenta"] for i in "123"], [0, 1, 1])
        self.assertEqual([ops[i]["firma"] for i in "123"], [1, 0, -1])  # válida, sin firma, alterada
        self.assertEqual(ops["2"]["frecuencia"], 4)
        self.assertNotIn("ana@correo.com", str(datos))  # el usuario va como seudónimo

    def test_sin_mongodb_no_se_rompe(self):
        from unittest import mock
        from appresso_food.antifraude import fraud_routes
        with mock.patch.object(fraud_routes, "get_service", side_effect=RuntimeError("caído")):
            datos = self.ia.get("/api/ia/operaciones").json()
        self.assertEqual((datos["ok"], datos["operaciones"]), (False, []))

    def test_demanda_simulada_si_hay_pocos_dias_y_pagina_con_librerias_verificadas(self):
        demanda = self.ia.get("/api/ia/demanda").json()
        self.assertEqual(len(demanda["simulado"]), 90)
        self.assertEqual(demanda["simulado"], self.ia.get("/api/ia/demanda").json()["simulado"])  # reproducible
        html = self.ia.get("/ia").text
        self.assertIn("@tensorflow/tfjs@4.22.0", html)
        self.assertIn("simple-statistics@7.8.8", html)
        self.assertEqual(html.count('integrity="sha384-'), 2)


if __name__ == "__main__":
    unittest.main()
