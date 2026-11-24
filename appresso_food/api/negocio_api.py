"""Módulos operativos del final: categorías y usuarios con roles (CRUD), facturas, pagos con comprobante,
integridad (hash encadenado) y reporte de calidad de datos.

Las escrituras públicas siguen las mismas reglas que /api/productos: lo sembrado (categorías del menú, usuarios del
equipo) no se borra desde afuera, y anular una factura exige el token del equipo del local (como cocina).
"""

import re
import sqlite3
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from appresso_food.api import gestos_api
from appresso_food.api.api_helpers import error_response, json_body, read_json_object
from appresso_food.nucleo import facturacion, seed_data, storage
from appresso_food.nucleo.limpieza import clave, nombre_propio, normalizar_texto, reporte_calidad

router = APIRouter()

ROLES = ("admin", "cajero", "cocina", "repartidor", "cliente")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.I)
USUARIOS_SEMBRADOS = {email for _, email, _ in seed_data.USERS}
# Renombrar una categoría arrastra a sus productos: las del menú solo se editan o borran desde el equipo del local.
CATEGORIAS_MENU = {clave(categoria) for _, categoria, _ in seed_data.PRODUCTS}


# --- Calidad e integridad ----------------------------------------------------------------------

@router.get("/api/calidad")
def quality_report():
    """Duplicados, precios inválidos, categorías inconsistentes, clientes repetidos y stock negativo."""
    return reporte_calidad(storage.get_products(), storage.get_all_orders(), storage.get_ingredients())


@router.get("/api/integridad")
def integrity():
    """Recorre las cadenas de hashes de facturas y pagos y revisa los comprobantes guardados."""
    estado = facturacion.integridad()
    estado["ok"] = estado["facturas"]["ok"] and estado["pagos"]["ok"] and not estado["comprobantes_con_problema"]
    return estado


# --- Categorías --------------------------------------------------------------------------------

def _categoria_publica(fila) -> dict:
    return {"id": fila[0], "nombre": fila[1], "descripcion": fila[2], "productos": fila[3]}


def _buscar_categoria(categoria_id: int):
    fila = next((c for c in storage.get_categories() if c[0] == categoria_id), None)
    if fila is None:
        raise HTTPException(404, f"No existe la categoría con id {categoria_id}. Consulte GET /api/categorias.")
    return fila


async def _leer_categoria(request: Request):
    data, error = await read_json_object(request)
    if error:
        return None, error
    nombre = normalizar_texto(data.get("nombre") or "")
    if not 2 <= len(nombre) <= 60 or not any(c.isalpha() for c in nombre):
        return None, error_response(422, "INVALID_NAME", "'nombre' es obligatorio: entre 2 y 60 caracteres, con letras.")
    descripcion = data.get("descripcion")
    if descripcion is not None and not isinstance(descripcion, str):
        return None, error_response(422, "INVALID_TYPE", "'descripcion' debe ser texto.")
    return (nombre, normalizar_texto(descripcion)[:200] if descripcion else None), None


def _categoria_repetida(nombre: str, excepto: Optional[int] = None):
    """Otra categoría que es la misma una vez normalizada ("postres " y "Postrés"), o una respuesta 409."""
    repetida = next((c for c in storage.get_categories() if clave(c[1]) == clave(nombre) and c[0] != excepto), None)
    if repetida:
        return error_response(409, "CATEGORIA_DUPLICADA", f"Ya existe la categoría '{repetida[1]}' (id {repetida[0]}).")
    return None


@router.get("/api/categorias")
def list_categories():
    return [_categoria_publica(c) for c in storage.get_categories()]


@router.post("/api/categorias", status_code=201, openapi_extra=json_body({"nombre": "Postres", "descripcion": "Dulces"}))
async def create_category(request: Request):
    datos, error = await _leer_categoria(request)
    if error:
        return error
    repetida = _categoria_repetida(datos[0])
    if repetida:
        return repetida
    nuevo = storage.create_category(*datos)
    return {"ok": True, "mensaje": f"Categoría creada con id {nuevo}.", "categoria": _categoria_publica(_buscar_categoria(nuevo))}


def _categoria_protegida(fila, request: Request):
    if clave(fila[1]) in CATEGORIAS_MENU and not gestos_api._directo_y_local(request):
        return error_response(403, "CATEGORIA_PROTEGIDA", f"'{fila[1]}' es una categoría del menú del local y no se modifica por la API pública. Cree su propia categoría con POST /api/categorias.")
    return None


@router.put("/api/categorias/{categoria_id}", openapi_extra=json_body({"nombre": "Postres", "descripcion": "Dulces"}))
async def update_category(categoria_id: int, request: Request):
    actual = _buscar_categoria(categoria_id)
    bloqueo = _categoria_protegida(actual, request)
    if bloqueo:
        return bloqueo
    datos, error = await _leer_categoria(request)
    if error:
        return error
    repetida = _categoria_repetida(datos[0], excepto=categoria_id)
    if repetida:
        return repetida
    storage.update_category(categoria_id, *datos)
    return {"ok": True, "mensaje": f"Categoría actualizada; {actual[3]} producto(s) quedaron con el nombre nuevo.",
            "categoria": _categoria_publica(_buscar_categoria(categoria_id))}


@router.delete("/api/categorias/{categoria_id}")
def delete_category(categoria_id: int, request: Request):
    actual = _buscar_categoria(categoria_id)
    bloqueo = _categoria_protegida(actual, request)
    if bloqueo:
        return bloqueo
    if actual[3]:
        return error_response(409, "CATEGORIA_EN_USO", f"'{actual[1]}' la usan {actual[3]} producto(s): cámbieles la categoría antes de borrarla.")
    storage.delete_category(categoria_id)
    return {"ok": True, "mensaje": f"Categoría '{actual[1]}' eliminada."}


# --- Usuarios con roles ------------------------------------------------------------------------

def _usuario_publico(fila, request: Request) -> dict:
    """Por la URL pública el correo va enmascarado (an***@correo.com): es un dato personal."""
    email = fila[2]
    if not gestos_api._directo_y_local(request):
        usuario, _, dominio = email.partition("@")
        email = f"{usuario[:2]}***@{dominio}"
    return {"id": fila[0], "nombre": fila[1], "email": email, "rol": fila[3], "activo": bool(fila[4]), "creado": fila[5]}


def _buscar_usuario(usuario_id: int):
    fila = storage.get_user(usuario_id)
    if fila is None:
        raise HTTPException(404, f"No existe el usuario con id {usuario_id}. Consulte GET /api/usuarios.")
    return fila


def _validar_usuario(data: dict, base: Optional[dict] = None):
    """Valida y limpia; con `base` (PATCH) solo exige los campos que llegan."""
    datos = dict(base or {})
    errores = []
    if "nombre" in data or base is None:
        nombre = nombre_propio(data.get("nombre") or "")
        if not 2 <= len(nombre) <= 80 or not any(c.isalpha() for c in nombre):
            errores.append("'nombre' es obligatorio: entre 2 y 80 caracteres, con letras.")
        datos["nombre"] = nombre
    if "email" in data or base is None:
        email = str(data.get("email") or "").strip().lower()
        if not EMAIL_RE.match(email):
            errores.append("'email' no es un correo válido (ej: ana@correo.com).")
        datos["email"] = email
    if "rol" in data or base is None:
        rol = str(data.get("rol") or "").strip().lower()
        if rol not in ROLES:
            errores.append(f"'rol' debe ser uno de: {', '.join(ROLES)}.")
        datos["rol"] = rol
    if "activo" in data:
        if not isinstance(data["activo"], bool):
            errores.append("'activo' debe ser true o false.")
        datos["activo"] = data["activo"]
    datos.setdefault("activo", True)
    return datos, errores


def _proteger_sembrado(fila, request: Request):
    if fila[2] in USUARIOS_SEMBRADOS and not gestos_api._directo_y_local(request):
        return error_response(403, "USUARIO_PROTEGIDO", f"{fila[1]} es del equipo del local y no se modifica por la API pública. Cree su propio usuario con POST /api/usuarios.")
    return None


def _guardar_usuario(usuario_id: Optional[int], datos: dict):
    try:
        if usuario_id is None:
            usuario_id = storage.create_user(datos["nombre"], datos["email"], datos["rol"], datos["activo"])
        else:
            storage.update_user(usuario_id, datos["nombre"], datos["email"], datos["rol"], datos["activo"])
    except sqlite3.IntegrityError:
        return None, error_response(409, "EMAIL_DUPLICADO", f"Ya hay un usuario con el correo {datos['email']}.")
    return usuario_id, None


EJEMPLO_USUARIO = {"nombre": "Ana López", "email": "ana@correo.com", "rol": "cajero", "activo": True}


@router.get("/api/usuarios")
def list_users(request: Request, rol: Optional[str] = None):
    usuarios = [_usuario_publico(u, request) for u in storage.get_users()]
    return [u for u in usuarios if not rol or u["rol"] == rol]


@router.get("/api/usuarios/{usuario_id}")
def get_user(usuario_id: int, request: Request):
    return _usuario_publico(_buscar_usuario(usuario_id), request)


@router.post("/api/usuarios", status_code=201, openapi_extra=json_body(EJEMPLO_USUARIO))
async def create_user(request: Request):
    data, error = await read_json_object(request)
    if error:
        return error
    datos, errores = _validar_usuario(data)
    if errores:
        return error_response(422, "DATOS_INVALIDOS", "; ".join(errores), formato_esperado=EJEMPLO_USUARIO | {"roles": ROLES})
    nuevo, error = _guardar_usuario(None, datos)
    if error:
        return error
    return {"ok": True, "mensaje": f"Usuario creado con id {nuevo}.", "usuario": _usuario_publico(_buscar_usuario(nuevo), request)}


async def _modificar_usuario(usuario_id: int, request: Request, parcial: bool):
    actual = _buscar_usuario(usuario_id)
    bloqueo = _proteger_sembrado(actual, request)
    if bloqueo:
        return bloqueo
    data, error = await read_json_object(request)
    if error:
        return error
    base = {"nombre": actual[1], "email": actual[2], "rol": actual[3], "activo": bool(actual[4])} if parcial else None
    datos, errores = _validar_usuario(data, base)
    if errores:
        return error_response(422, "DATOS_INVALIDOS", "; ".join(errores), formato_esperado=EJEMPLO_USUARIO | {"roles": ROLES})
    _, error = _guardar_usuario(usuario_id, datos)
    if error:
        return error
    return {"ok": True, "mensaje": f"Usuario {usuario_id} actualizado.", "usuario": _usuario_publico(_buscar_usuario(usuario_id), request)}


@router.put("/api/usuarios/{usuario_id}", openapi_extra=json_body(EJEMPLO_USUARIO))
async def replace_user(usuario_id: int, request: Request):
    return await _modificar_usuario(usuario_id, request, parcial=False)


@router.patch("/api/usuarios/{usuario_id}", openapi_extra=json_body({"rol": "cocina"}))
async def patch_user(usuario_id: int, request: Request):
    return await _modificar_usuario(usuario_id, request, parcial=True)


@router.delete("/api/usuarios/{usuario_id}")
def delete_user(usuario_id: int, request: Request):
    actual = _buscar_usuario(usuario_id)
    bloqueo = _proteger_sembrado(actual, request)
    if bloqueo:
        return bloqueo
    storage.delete_user(usuario_id)
    return {"ok": True, "mensaje": f"Usuario {actual[1]} eliminado."}


# --- Facturas, pagos y comprobantes ------------------------------------------------------------

@router.get("/api/facturas")
def list_invoices():
    return facturacion.con_estado(storage.get_invoices())


@router.get("/api/facturas/{numero}")
def get_invoice(numero: str):
    factura = next((f for f in list_invoices() if f["number"] == numero.upper()), None)
    if factura is None:
        raise HTTPException(404, f"No existe la factura {numero}.")
    return factura


@router.post("/api/facturas/{numero}/anular", openapi_extra=json_body({"motivo": "El cliente canceló", "token": "token del equipo"}))
async def annul_invoice(numero: str, request: Request):
    """Anular es sensible (es como se esconde un fraude): solo con el token del equipo, como atender en cocina."""
    data, error = await read_json_object(request)
    if error:
        return error
    if not gestos_api.token_valido(request, data.get("token")):
        return error_response(403, "SOLO_EQUIPO", "Las facturas solo se anulan desde el equipo del local (página /facturas).")
    try:
        nota = facturacion.anular_factura(numero.upper(), str(data.get("motivo") or ""), str(data.get("anulada_por") or "") or None)
    except facturacion.FacturaError as exc:
        return error_response(409, "NO_SE_PUEDE_ANULAR", str(exc))
    return {"ok": True, "mensaje": f"Factura {numero.upper()} anulada con la nota crédito {nota['number']}.", "nota_credito": nota}


@router.get("/api/pagos")
def list_payments():
    return storage.get_payments()


@router.post("/api/pedidos/{order_id}/comprobante")
async def upload_receipt(order_id: int, archivo: UploadFile = File(...), clave_pedido: str = Form("", alias="clave")):
    """Comprobante de pago: solo con la clave de la confirmación del pedido, y solo si el pedido ya está pagado."""
    order = storage.get_order(order_id)
    if order is None:
        raise HTTPException(404, f"No existe el pedido #{order_id}.")
    if not order.clave_valida(clave_pedido):
        return error_response(403, "SIN_PERMISO", "El comprobante solo se adjunta desde la confirmación del pedido.")
    if not order.paid_txn:
        return error_response(409, "SIN_PAGO", "Primero hay que pagar el pedido; después se adjunta el comprobante.")
    contenido = await archivo.read(facturacion.MAX_COMPROBANTE + 1)
    try:
        fila = facturacion.registrar_comprobante(order_id, archivo.filename or "comprobante", contenido)
    except facturacion.FacturaError as exc:
        return error_response(422, "COMPROBANTE_INVALIDO", str(exc))
    return {"ok": True, "mensaje": "Comprobante guardado y encadenado.", "sha256": fila["receipt_sha256"], "factura": fila["invoice_number"]}
