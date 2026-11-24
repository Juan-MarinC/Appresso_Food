// /ia: clasificador de operaciones con TensorFlow.js (API de capas, la de Keras) y anomalías y demanda con
// Simple Statistics. Todo corre en el navegador; el servidor solo entrega los datos (/api/ia/...).
(function () {
    "use strict";
    const $ = (id) => document.getElementById(id);
    const esc = window.Fraud.esc;  // escape HTML compartido (fraud-common.js)
    const pesos = (n) => "$" + Math.round(n).toLocaleString("es-CO");
    const pct = (x) => (100 * x).toFixed(1) + " %";
    const METODOS = ["Tarjeta", "Nequi", "Crédito"];  // el resto cuenta como "Otro"
    const MODELO_URL = "indexeddb://appresso-fraude";
    const CLAVE_NORMALIZACION = "appresso-ia-normalizacion";
    const CLAVE_METRICAS = "appresso-ia-metricas";
    let operaciones = [];
    let modelo = null;
    let normalizacion = null;  // { media, desv } como tensores

    const guardar = (clave, valor) => { try { localStorage.setItem(clave, JSON.stringify(valor)); } catch (e) { /* sin almacenamiento */ } };
    const leer = (clave) => { try { return JSON.parse(localStorage.getItem(clave)); } catch (e) { return null; } };
    const error = (texto) => { $("ia-error").hidden = false; $("ia-error").textContent = texto; };
    const estado = (texto) => { $("ia-estado").textContent = texto; };

    // Azar con semilla: el mismo reparto entrenamiento/prueba en cada corrida (resultados comparables).
    function azar(semilla) {
        return function () {
            semilla = (semilla + 0x6D2B79F5) | 0;
            let t = Math.imul(semilla ^ (semilla >>> 15), 1 | semilla);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    // --- Entradas del modelo: lo que pide el enunciado (fecha, monto, método, firma, frecuencia, usuario) ---
    function rasgos(o) {
        const fecha = new Date(o.fecha);
        const hora = fecha.getHours() + fecha.getMinutes() / 60;
        const dia = fecha.getDay();
        const m = METODOS.indexOf(o.metodo);
        return [
            Math.log1p(o.monto),                                              // monto (escala logarítmica)
            Math.sin(2 * Math.PI * hora / 24), Math.cos(2 * Math.PI * hora / 24),  // hora como círculo: 23 h queda junto a 0 h
            dia === 0 || dia === 6 ? 1 : 0,                                   // fin de semana
            m === 0 ? 1 : 0, m === 1 ? 1 : 0, m === 2 ? 1 : 0, m === -1 ? 1 : 0,  // método de pago (one-hot)
            o.firma,                                                          // 1 válida, 0 sin firma, -1 alterada
            Math.log1p(o.frecuencia),                                         // operaciones del usuario en la ventana de 10 s
            Math.log1p(o.operaciones_usuario || 1),                           // historial del usuario
        ];
    }

    function normalizar(x) {
        return tf.tidy(() => x.sub(normalizacion.media).div(normalizacion.desv));
    }

    // Separa 20 % de cada clase para prueba (estratificado), siempre igual gracias a la semilla.
    function repartir() {
        const rnd = azar(42);
        const entrenamiento = [], prueba = [];
        for (const clase of [0, 1]) {
            const grupo = operaciones.filter((o) => o.fraudulenta === clase).map((o) => [rnd(), o]).sort((a, b) => a[0] - b[0]).map((p) => p[1]);
            const corte = Math.round(grupo.length * 0.2);
            prueba.push(...grupo.slice(0, corte));
            entrenamiento.push(...grupo.slice(corte));
        }
        return { entrenamiento, prueba };
    }

    // Aumento de datos: copias de las fraudulentas con ruido gaussiano hasta que sean ~30 % del entrenamiento.
    function aumentar(x, y) {
        const etiquetas = y.dataSync();
        const fraudes = [];
        etiquetas.forEach((v, i) => { if (v === 1) fraudes.push(i); });
        const normales = etiquetas.length - fraudes.length;
        const faltan = Math.round(0.3 * normales / 0.7) - fraudes.length;
        if (!fraudes.length || faltan <= 0) return { x, y, agregadas: 0 };
        return tf.tidy(() => {
            const indices = tf.tensor1d(Array.from({ length: faltan }, (_, i) => fraudes[i % fraudes.length]), "int32");
            const copias = tf.gather(x, indices).add(tf.randomNormal([faltan, x.shape[1]], 0, 0.05, "float32", 7));
            return { x: tf.concat([x, copias]), y: tf.concat([y, tf.ones([faltan, 1])]), agregadas: faltan };
        });
    }

    function crearModelo(entradas) {
        const m = tf.sequential();
        m.add(tf.layers.dense({ inputShape: [entradas], units: 16, activation: "relu" }));
        m.add(tf.layers.dropout({ rate: 0.2 }));
        m.add(tf.layers.dense({ units: 8, activation: "relu" }));
        m.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
        m.compile({ optimizer: tf.train.adam(0.01), loss: "binaryCrossentropy", metrics: ["accuracy"] });
        return m;
    }

    function probabilidades(lista) {
        if (!lista.length) return [];
        return tf.tidy(() => Array.from(modelo.predict(normalizar(tf.tensor2d(lista.map(rasgos)))).dataSync()));
    }

    function graficarPerdida(perdidas) {
        const svg = $("ia-perdida");
        const max = Math.max(...perdidas, 0.01);
        const puntos = perdidas.map((p, i) => `${20 + i * (560 / Math.max(perdidas.length - 1, 1))},${125 - (p / max) * 110}`).join(" ");
        svg.innerHTML = `<polyline points="${puntos}" class="ia-linea"/><text x="20" y="12" class="ia-texto">pérdida por época: ${perdidas[perdidas.length - 1].toFixed(4)}</text>`;
    }

    async function entrenar() {
        const fraudes = operaciones.filter((o) => o.fraudulenta).length;
        if (fraudes < 5) { estado("Hacen falta al menos 5 operaciones fraudulentas para aprender (envíe ráfagas a la pasarela)."); return; }
        $("btn-entrenar").disabled = true;
        const { entrenamiento, prueba } = repartir();
        const crudo = tf.tensor2d(entrenamiento.map(rasgos));
        const { mean, variance } = tf.moments(crudo, 0);
        if (normalizacion) { normalizacion.media.dispose(); normalizacion.desv.dispose(); }
        normalizacion = { media: mean, desv: tf.tidy(() => variance.sqrt().add(1e-6)) };
        variance.dispose();
        const x = normalizar(crudo);
        const y = tf.tensor2d(entrenamiento.map((o) => [o.fraudulenta]));
        crudo.dispose();
        const aumentado = aumentar(x, y);
        if (modelo) modelo.dispose();
        modelo = crearModelo(x.shape[1]);
        const perdidas = [];
        const EPOCAS = 25;
        await modelo.fit(aumentado.x, aumentado.y, {
            epochs: EPOCAS, batchSize: 128, shuffle: true,
            callbacks: {
                onEpochEnd: async (epoca, logs) => {
                    perdidas.push(logs.loss);
                    graficarPerdida(perdidas);
                    estado(`Entrenando… época ${epoca + 1} de ${EPOCAS} · ${entrenamiento.length} operaciones + ${aumentado.agregadas} sintéticas (aumento de datos)`);
                    await tf.nextFrame();
                },
            },
        });
        if (aumentado.x !== x) { aumentado.x.dispose(); aumentado.y.dispose(); }
        x.dispose(); y.dispose();
        evaluar(prueba);
        await guardarModelo();
        mostrarClasificacion();
        $("btn-entrenar").disabled = false;
        $("btn-descargar").disabled = false;
        estado(`Listo. Modelo entrenado con ${entrenamiento.length} operaciones y guardado en este navegador.`);
    }

    function evaluar(prueba) {
        const p = probabilidades(prueba);
        let vp = 0, vn = 0, fp = 0, fn = 0;
        prueba.forEach((o, i) => {
            const predijo = p[i] >= 0.5 ? 1 : 0;
            if (predijo && o.fraudulenta) vp++; else if (predijo) fp++; else if (o.fraudulenta) fn++; else vn++;
        });
        const n = prueba.length;
        const m = {
            accuracy: (vp + vn) / n, precision: vp / Math.max(vp + fp, 1), recall: vp / Math.max(vp + fn, 1),
            base: (vn + fp) / n, vp, vn, fp, fn, prueba: n, fecha: new Date().toISOString(),
        };
        m.f1 = m.precision + m.recall ? 2 * m.precision * m.recall / (m.precision + m.recall) : 0;
        mostrarMetricas(m);
        guardar(CLAVE_METRICAS, m);  // el panel muestra estas métricas
    }

    function mostrarMetricas(m) {
        $("ia-resultados").hidden = false;
        [["m-vn", m.vn], ["m-fp", m.fp], ["m-fn", m.fn], ["m-vp", m.vp]].forEach(([id, v]) => { $(id).textContent = v; });
        $("ia-accuracy").textContent = pct(m.accuracy);
        $("ia-recall").textContent = pct(m.recall);
        $("ia-metricas").innerHTML = `Precisión global <strong>${pct(m.accuracy)}</strong> · precisión en fraude <strong>${pct(m.precision)}</strong> ·
            sensibilidad <strong>${pct(m.recall)}</strong> · F1 <strong>${m.f1.toFixed(3)}</strong> sobre ${m.prueba} operaciones de prueba.
            Ojo: decir siempre "normal" ya acertaría ${pct(m.base)}; por eso importa cuántos fraudes detecta (sensibilidad).`;
    }

    async function guardarModelo() {
        await modelo.save(MODELO_URL);  // IndexedDB del navegador
        guardar(CLAVE_NORMALIZACION, { media: Array.from(normalizacion.media.dataSync()), desv: Array.from(normalizacion.desv.dataSync()) });
        $("btn-cargar").disabled = false;
    }

    async function cargarModelo() {
        const guardada = leer(CLAVE_NORMALIZACION);
        try {
            if (!guardada) throw new Error("sin normalización");
            if (modelo) modelo.dispose();
            modelo = await tf.loadLayersModel(MODELO_URL);
            normalizacion = { media: tf.tensor1d(guardada.media), desv: tf.tensor1d(guardada.desv) };
        } catch (e) {
            estado("No hay un modelo guardado en este navegador: entrénelo primero.");
            return;
        }
        const m = leer(CLAVE_METRICAS);
        if (m) mostrarMetricas(m);
        mostrarClasificacion();
        $("btn-descargar").disabled = false;
        estado("Modelo cargado desde este navegador (IndexedDB), sin volver a entrenar.");
    }

    function mostrarClasificacion() {
        const recientes = operaciones.slice(-25).reverse();
        const p = probabilidades(recientes);
        const firma = { 1: "válida", 0: "—", "-1": "alterada" };
        $("ia-tabla-titulo").hidden = false;
        $("ia-tabla").innerHTML = `<tr><th>Fecha</th><th>Operación</th><th>Monto</th><th>Método</th><th>Firma</th><th>Frecuencia</th><th>Prob. fraude</th><th>Modelo</th><th>Antifraude</th></tr>` +
            recientes.map((o, i) => {
                const fraude = p[i] >= 0.5;
                return `<tr><td>${esc(o.fecha.replace("T", " "))}</td><td><code>${esc(o.id)}</code></td><td>${pesos(o.monto)}</td><td>${esc(o.metodo)}</td>
                    <td>${firma[o.firma]}</td><td>${o.frecuencia}</td>
                    <td><span class="ia-barra" style="--p:${(100 * p[i]).toFixed(0)}%"></span> ${pct(p[i])}</td>
                    <td><strong class="${fraude ? "ia-fraude" : "ia-normal"}">${fraude ? "Fraudulenta" : "Normal"}</strong></td>
                    <td>${o.fraudulenta ? "Fraudulenta" : "Normal"} ${(o.fraudulenta === 1) === fraude ? "✓" : "✗"}</td></tr>`;
            }).join("");
    }

    function probarOperacion(e) {
        e.preventDefault();
        if (!modelo) { $("p-resultado").textContent = "Primero entrene o cargue el modelo."; return; }
        const hoy = new Date();
        hoy.setHours(Number($("p-hora").value) || 0, 0, 0, 0);
        const op = {
            fecha: hoy.toISOString(), monto: Number($("p-monto").value) || 1, metodo: $("p-metodo").value,
            firma: Number($("p-firma").value), frecuencia: Number($("p-frecuencia").value) || 1,
        };
        op.operaciones_usuario = op.frecuencia;
        const [prob] = probabilidades([op]);
        $("p-resultado").innerHTML = `Probabilidad de fraude: <strong>${pct(prob)}</strong> → <strong class="${prob >= 0.5 ? "ia-fraude" : "ia-normal"}">${prob >= 0.5 ? "fraudulenta" : "normal"}</strong>`;
    }

    // --- Simple Statistics: perfil estadístico y anomalías por usuario ---
    function anomaliasUsuarios() {
        const montos = operaciones.map((o) => o.monto);
        if (montos.length < 4) { $("ss-montos").textContent = "Faltan operaciones para calcular cuartiles."; return; }
        const media = ss.mean(montos), desv = ss.standardDeviation(montos) || 1;
        const [q1, q2, q3] = [0.25, 0.5, 0.75].map((p) => ss.quantile(montos, p));
        const ric = ss.interquartileRange(montos);
        const bajo = q1 - 1.5 * ric, alto = q3 + 1.5 * ric;
        const frecuencias = operaciones.map((o) => o.frecuencia);
        const mf = ss.mean(frecuencias), df = ss.standardDeviation(frecuencias) || 1;
        const atipicos = montos.filter((v) => v < bajo || v > alto).length;
        $("ss-montos").innerHTML = `Montos: media <strong>${pesos(media)}</strong>, desviación ${pesos(desv)} · Q1 ${pesos(q1)} · mediana ${pesos(q2)} · Q3 ${pesos(q3)} ·
            RIC ${pesos(ric)} · atípicos fuera de ${pesos(Math.max(bajo, 0))} – ${pesos(alto)}: <strong>${atipicos}</strong> de ${montos.length}.`;
        const usuarios = new Map();
        for (const o of operaciones) {
            const u = usuarios.get(o.usuario) || { usuario: o.usuario, montos: [], frecuencias: [] };
            u.montos.push(o.monto); u.frecuencias.push(o.frecuencia);
            usuarios.set(o.usuario, u);
        }
        const perfiles = [...usuarios.values()].map((u) => {
            const zMonto = ss.zScore(ss.max(u.montos), media, desv);
            const zFrec = ss.zScore(ss.max(u.frecuencias), mf, df);
            const fuera = u.montos.filter((v) => v < bajo || v > alto).length;
            const puntaje = Math.abs(zMonto) + Math.abs(zFrec);
            return { ...u, n: u.montos.length, media: ss.mean(u.montos), zMonto, zFrec, fuera, puntaje, alerta: puntaje > 3 || fuera > 0 };
        }).sort((a, b) => b.puntaje - a.puntaje).slice(0, 10);
        const alertas = perfiles.filter((p) => p.alerta).length;
        $("ss-usuarios").innerHTML = `<tr><th>Usuario (seudónimo)</th><th>Operaciones</th><th>Monto prom.</th><th>z monto</th><th>z frecuencia</th><th>Puntaje</th><th>Alerta</th></tr>` +
            perfiles.map((p) => `<tr><td><code>${esc(p.usuario)}</code></td><td>${p.n}</td><td>${pesos(p.media)}</td><td>${p.zMonto.toFixed(2)}</td><td>${p.zFrec.toFixed(2)}</td>
                <td><strong>${p.puntaje.toFixed(2)}</strong></td><td>${p.alerta ? `<span class="chip-alerta">${p.fuera ? "monto atípico" : "comportamiento anormal"}</span>` : "—"}</td></tr>`).join("") +
            `<tr><td colspan="7"><small>Los 10 usuarios con mayor puntaje · ${alertas} con alerta</small></td></tr>`;
    }

    async function anomaliasEmpleados() {
        const facturas = await (await fetch("/api/facturas")).json();
        const grupos = new Map();
        for (const f of facturas) {
            const quien = f.issued_by || "Kiosko / cliente";
            const g = grupos.get(quien) || { quien, descuentos: [], anulaciones: 0 };
            if (f.kind === "FACTURA") g.descuentos.push(f.discount_pct); else g.anulaciones++;
            grupos.set(quien, g);
        }
        const filas = [...grupos.values()].map((g) => ({ ...g, facturas: g.descuentos.length, promedio: g.descuentos.length ? ss.mean(g.descuentos) : 0 }));
        if (!filas.length) { $("ss-empleados").innerHTML = `<tr><td class="empty">Aún no hay facturas.</td></tr>`; return; }
        const promedios = filas.map((f) => f.promedio);
        const alto = ss.quantile(promedios, 0.75) + 1.5 * ss.interquartileRange(promedios);
        $("ss-empleados").innerHTML = `<tr><th>Quién factura</th><th>Facturas</th><th>Descuento prom.</th><th>Anulaciones</th><th>Alerta</th></tr>` +
            filas.map((f) => {
                const alertas = [];
                if (f.promedio > alto && f.promedio > 0) alertas.push("descuentos atípicos");
                if (f.facturas && f.anulaciones / f.facturas > 0.5) alertas.push("muchas anulaciones");
                return `<tr><td>${esc(f.quien)}</td><td>${f.facturas}</td><td>${f.promedio.toFixed(1)} %</td><td>${f.anulaciones}</td>
                    <td>${alertas.length ? `<span class="chip-alerta">${alertas.join(", ")}</span>` : "—"}</td></tr>`;
            }).join("");
    }

    // --- Demanda: regresión lineal (Simple Statistics) contra una red con ventana de 7 días (TensorFlow) ---
    const VENTANA = 7, FUTURO = 7, PRUEBA = 14;

    async function redDemanda(serie) {
        const escala = Math.max(...serie) || 1;
        const s = serie.map((v) => v / escala);
        const xs = [], ys = [];
        for (let i = VENTANA; i < s.length; i++) { xs.push(s.slice(i - VENTANA, i)); ys.push([s[i]]); }
        const m = tf.sequential();
        m.add(tf.layers.dense({ inputShape: [VENTANA], units: 16, activation: "relu" }));
        m.add(tf.layers.dense({ units: 1 }));
        m.compile({ optimizer: tf.train.adam(0.01), loss: "meanSquaredError" });
        const x = tf.tensor2d(xs), y = tf.tensor2d(ys);
        await m.fit(x, y, { epochs: 200, batchSize: 16, verbose: 0 });
        x.dispose(); y.dispose();
        // Predice día a día: cada predicción entra en la ventana del día siguiente.
        return function (ultimos, pasos) {
            const ventana = ultimos.slice(-VENTANA).map((v) => v / escala), salida = [];
            for (let k = 0; k < pasos; k++) {
                const siguiente = tf.tidy(() => m.predict(tf.tensor2d([ventana])).dataSync()[0]);
                salida.push(siguiente * escala);
                ventana.shift(); ventana.push(siguiente);
            }
            return salida;
        };
    }

    async function demanda() {
        const d = await (await fetch("/api/ia/demanda")).json();
        const serie = d.simulado || d.reales;
        $("dem-origen").innerHTML = d.simulado
            ? `Hay pocos días reales con ventas (${d.reales.filter((v) => v).length}), así que se usa un <strong>histórico simulado de 90 días</strong> (tendencia creciente y más ventas viernes y sábado).`
            : `Unidades vendidas por día (${serie.length} días reales).`;
        const entreno = serie.slice(0, -PRUEBA), real = serie.slice(-PRUEBA);
        // Simple Statistics: y = m·x + b sobre los días de entrenamiento
        const lineal = ss.linearRegressionLine(ss.linearRegression(entreno.map((y, x) => [x, y])));
        const r2 = ss.rSquared(entreno.map((y, x) => [x, y]), lineal);
        const predecirRed = await redDemanda(entreno);
        const predLineal = real.map((_, i) => lineal(entreno.length + i));
        const predRed = predecirRed(entreno, PRUEBA);
        const mae = (pred) => ss.mean(pred.map((p, i) => Math.abs(p - real[i])));
        const errLineal = mae(predLineal), errRed = mae(predRed);
        const gana = errRed < errLineal ? "la red neuronal" : "la regresión lineal";
        $("dem-error").innerHTML = `Error medio en los últimos ${PRUEBA} días (que ningún modelo vio): regresión lineal <strong>${errLineal.toFixed(1)}</strong> unidades,
            red neuronal <strong>${errRed.toFixed(1)}</strong> unidades: aquí gana <strong>${gana}</strong>. R² de la recta: ${r2.toFixed(3)}.
            La recta solo sigue la tendencia; la red mira los últimos 7 días, por eso sus predicciones suben los fines de semana.`;
        const futLineal = Array.from({ length: FUTURO }, (_, i) => lineal(serie.length + i));
        const futRed = predecirRed(serie, FUTURO);
        $("dem-tabla").innerHTML = `<tr><th>Día</th>${futLineal.map((_, i) => `<th>+${i + 1}</th>`).join("")}</tr>
            <tr><th>Regresión lineal (Simple Statistics)</th>${futLineal.map((v) => `<td>${Math.round(v)}</td>`).join("")}</tr>
            <tr><th>Red neuronal (TensorFlow)</th>${futRed.map((v) => `<td>${Math.round(v)}</td>`).join("")}</tr>`;
        graficarDemanda(serie, futLineal, futRed, lineal);
    }

    function graficarDemanda(serie, futLineal, futRed, lineal) {
        const total = serie.length + FUTURO;
        const max = Math.max(...serie, ...futLineal, ...futRed) * 1.1;
        const px = (i) => 20 + i * (560 / (total - 1)), py = (v) => 185 - (v / max) * 170;
        const linea = (valores, desde, clase) => `<polyline class="${clase}" points="${valores.map((v, i) => `${px(desde + i)},${py(v)}`).join(" ")}"/>`;
        $("dem-grafica").innerHTML = linea(serie, 0, "ia-linea ia-linea--gris") +
            `<polyline class="ia-linea ia-linea--recta" points="${px(0)},${py(lineal(0))} ${px(total - 1)},${py(lineal(total - 1))}"/>` +
            linea([serie[serie.length - 1], ...futRed], serie.length - 1, "ia-linea ia-linea--red") +
            `<line x1="${px(serie.length - 1)}" y1="10" x2="${px(serie.length - 1)}" y2="190" class="ia-hoy"/>` +
            `<text x="24" y="14" class="ia-texto">— unidades por día   - - recta (Simple Statistics)   — red neuronal (TensorFlow)</text>`;
    }

    // --- Arranque ---
    async function iniciar() {
        if (typeof tf === "undefined" || typeof ss === "undefined") {
            error("No se pudieron cargar TensorFlow.js o Simple Statistics (se descargan de internet). Revise la conexión y recargue.");
            estado("");
            return;
        }
        const datos = await (await fetch("/api/ia/operaciones")).json();
        if (!datos.ok) error(datos.mensaje);
        operaciones = datos.operaciones || [];
        $("ia-total").textContent = operaciones.length.toLocaleString("es-CO");
        $("ia-fraude").textContent = `${(datos.fraudulentas || 0).toLocaleString("es-CO")} (${pct((datos.fraudulentas || 0) / Math.max(operaciones.length, 1))})`;
        $("btn-entrenar").disabled = false;
        $("btn-cargar").disabled = !leer(CLAVE_NORMALIZACION);
        estado(`${operaciones.length} operaciones listas. Pulse "Entrenar" (tarda unos segundos).`);
        anomaliasUsuarios();
        anomaliasEmpleados().catch(() => {});
        demanda().catch((e) => { $("dem-error").textContent = "No se pudo calcular la demanda: " + e.message; });
    }

    $("btn-entrenar").addEventListener("click", () => entrenar().catch((e) => { error("Falló el entrenamiento: " + e.message); $("btn-entrenar").disabled = false; }));
    $("btn-cargar").addEventListener("click", () => cargarModelo());
    $("btn-descargar").addEventListener("click", () => modelo && modelo.save("downloads://appresso-fraude"));
    $("ia-probar").addEventListener("submit", probarOperacion);
    iniciar().catch((e) => error("No se pudieron cargar los datos: " + e.message));
})();
