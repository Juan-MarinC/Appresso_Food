/* Utilidades compartidas por /transacciones y /antifraude. Sin dependencias externas. */
(function () {
    const $ = (id) => document.getElementById(id);

    function esc(value) {
        return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
    }

    function money(n) {
        const value = Number(n || 0);
        return (value < 0 ? "-$" : "$") + Math.abs(value).toLocaleString("es-CO", { maximumFractionDigits: 0 });
    }

    function time(iso) {
        if (!iso) return "—";
        const d = new Date(iso);
        if (isNaN(d)) return esc(iso);
        const p = (n, l = 2) => String(n).padStart(l, "0");
        return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}.${p(d.getMilliseconds(), 3)}`;
    }

    function datetime(iso) {
        if (!iso) return "—";
        const d = new Date(iso);
        if (isNaN(d)) return esc(iso);
        return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")} ${time(iso)}`;
    }

    async function api(path, options) {
        const response = await fetch(path, options);
        let data = null;
        try { data = await response.json(); } catch (_) { /* respuesta sin JSON */ }
        return { status: response.status, ok: response.ok, data };
    }

    /**
     * Dibuja una ventana deslizante como línea de tiempo SVG.
     * El borde derecho es la transacción más reciente; el izquierdo, N segundos antes.
     * entries: [{idTxn, fecha(ISO), valor}]  - windowSeconds - limit
     */
    // Ventana deslizante de un usuario: barra de tiempo (de "hace N s" a "ahora") con sus transacciones como puntos.
    // En HTML y no en SVG: así textos y puntos conservan su tamaño en cualquier ancho (panel, pasarela, celular).
    // Las que caen casi al mismo tiempo (una ráfaga) se agrupan en un círculo con el número; el detalle va en etiquetas.
    // (Se llama windowSvg por compatibilidad con quienes la usan.)
    function windowSvg(entries, windowSeconds, limit, anchorIso) {
        const list = (entries || []).map((e) => ({ ...e, t: new Date(e.fecha).getTime() })).sort((a, b) => a.t - b.t);
        const anchor = anchorIso ? new Date(anchorIso).getTime() : (list.length ? list[list.length - 1].t : Date.now());
        const pct = (t) => 100 * Math.max(0, Math.min(1, 1 - (anchor - t) / (windowSeconds * 1000)));
        const tono = list.length >= limit ? "hot" : (list.length === limit - 1 ? "warn" : "");
        const grupos = [];
        for (const e of list) {
            const x = pct(e.t), ultimo = grupos[grupos.length - 1];
            if (ultimo && x - ultimo.x < 3) { ultimo.items.push(e); ultimo.x = (ultimo.x * (ultimo.items.length - 1) + x) / ultimo.items.length; }
            else grupos.push({ x, items: [e] });
        }
        const paso = windowSeconds <= 10 ? 2 : windowSeconds <= 30 ? 5 : 10;
        const segundos = [];
        for (let s = windowSeconds; s >= 0; s -= paso) segundos.push(s);
        const marcas = segundos.slice(1, -1).map((s) => `<span class="wviz__marca" style="left:${100 * (1 - s / windowSeconds)}%"></span>`).join("");
        const puntos = grupos.map((g) => {
            const titulo = g.items.map((e) => `#${e.idTxn} · ${time(e.fecha)} · ${money(e.valor)}`).join("\n");
            const varios = g.items.length > 1;
            return `<span class="wviz__punto ${tono} ${varios ? "wviz__punto--grupo" : ""}" style="left:${g.x}%" title="${esc(titulo)}">${varios ? g.items.length : ""}</span>`;
        }).join("");
        const eje = segundos.map((s) => `<span>${s === 0 ? "ahora" : `hace ${s} s`}</span>`).join("");
        const recientes = list.slice().reverse();
        const etiquetas = recientes.slice(0, 5).map((e) => {
            const id = String(e.idTxn);
            const hace = Math.max(0, (anchor - e.t) / 1000);
            return `<span class="wchip"><b>#${esc(id.length > 14 ? "…" + id.slice(-11) : id)}</b> hace ${hace.toFixed(hace < 10 ? 1 : 0).replace(".", ",")} s · ${money(e.valor)}</span>`;
        }).join("");
        const mas = recientes.length > 5 ? `<span class="wchip wchip--mas">+${recientes.length - 5} más</span>` : "";
        return `<div class="wviz" role="img" aria-label="${list.length} transacciones en los últimos ${windowSeconds} segundos">
            <div class="wviz__pista ${tono}">${marcas}${puntos}</div>
            <div class="wviz__eje">${eje}</div>
            ${list.length ? `<div class="wviz__lista">${etiquetas}${mas}</div>` : ""}</div>`;
    }

    function chip(kind, value) {
        return `<span class="chip ${kind}-${esc(value)}">${esc(value)}</span>`;
    }

    window.Fraud = { $, esc, money, time, datetime, api, windowSvg, chip };
})();
