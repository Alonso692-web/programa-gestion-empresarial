from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import json

app = Flask(__name__)

class AnalisisEstadistico:
    def __init__(self):
        self.es_muestral = True
        # Tabla de constantes para gráficos de control. Fuente: Estándares de Ingeniería.
        self.CONTROL_CHART_CONSTANTS = {
            2: {'A2': 1.880, 'D3': 0, 'D4': 3.267}, 3: {'A2': 1.023, 'D3': 0, 'D4': 2.574},
            4: {'A2': 0.729, 'D3': 0, 'D4': 2.282}, 5: {'A2': 0.577, 'D3': 0, 'D4': 2.114},
            6: {'A2': 0.483, 'D3': 0, 'D4': 2.004}, 7: {'A2': 0.419, 'D3': 0.076, 'D4': 1.924},
            8: {'A2': 0.373, 'D3': 0.136, 'D4': 1.864}, 9: {'A2': 0.337, 'D3': 0.184, 'D4': 1.816},
            10: {'A2': 0.308, 'D3': 0.223, 'D4': 1.777}, 11: {'A2': 0.285, 'D3': 0.256, 'D4': 1.744},
            12: {'A2': 0.266, 'D3': 0.283, 'D4': 1.717}, 13: {'A2': 0.249, 'D3': 0.307, 'D4': 1.693},
            14: {'A2': 0.235, 'D3': 0.328, 'D4': 1.672}, 15: {'A2': 0.223, 'D3': 0.347, 'D4': 1.653}
        }

    def calcular_estadisticas_basicas(self, datos):
        datos = np.array(datos)
        media = float(np.mean(datos)); mediana = float(np.median(datos))
        valores_unicos, conteos = np.unique(datos, return_counts=True)
        max_count = np.max(conteos)
        modas = valores_unicos[conteos == max_count]
        if len(modas) == 1: moda = float(modas[0])
        elif len(modas) == len(valores_unicos): moda = "No hay moda"
        else: moda = [float(m) for m in modas]
        if self.es_muestral:
            varianza = float(np.var(datos, ddof=1)) if len(datos) > 1 else 0
            desviacion_std = float(np.std(datos, ddof=1)) if len(datos) > 1 else 0
        else:
            varianza = float(np.var(datos, ddof=0))
            desviacion_std = float(np.std(datos, ddof=0))
        return {
            'media': round(media, 4), 'mediana': round(mediana, 4), 'moda': moda,
            'varianza': round(varianza, 4), 'desviacion_estandar': round(desviacion_std, 4),
            'valor_minimo': round(float(np.min(datos)), 4), 'valor_maximo': round(float(np.max(datos)), 4),
            'rango': round(float(np.max(datos) - np.min(datos)), 4)
        }

    def calcular_analisis_subgrupos(self, df):
        num_muestras, tamano_muestra = df.shape
        if tamano_muestra not in self.CONTROL_CHART_CONSTANTS:
            raise ValueError(f"No hay constantes para tamaño de muestra n={tamano_muestra}. Tamaños soportados: 2-15.")
        
        # Calcular promedios y rangos para cada muestra
        promedios_x = df.mean(axis=1)
        rangos_r = df.max(axis=1) - df.min(axis=1)
        
        # Calcular líneas centrales
        gran_media_x_bar_bar = promedios_x.mean()
        rango_promedio_r_bar = rangos_r.mean()
        
        # Obtener constantes
        constantes = self.CONTROL_CHART_CONSTANTS[tamano_muestra]
        A2, D3, D4 = constantes['A2'], constantes['D3'], constantes['D4']
        
        # Calcular límites de control para la gráfica X
        lcs_x = gran_media_x_bar_bar + A2 * rango_promedio_r_bar
        lci_x = gran_media_x_bar_bar - A2 * rango_promedio_r_bar
        
        # Calcular límites de control para la gráfica R
        lcs_r = D4 * rango_promedio_r_bar
        lci_r = D3 * rango_promedio_r_bar
        
        return {
            "metricas": {
                'numero_muestras_k': num_muestras, 'tamano_muestra_n': tamano_muestra,
                'gran_media_x_bar_bar': round(gran_media_x_bar_bar, 4),
                'rango_promedio_r_bar': round(rango_promedio_r_bar, 4),
                'lcs_x': round(lcs_x, 4), 'lci_x': round(lci_x, 4),
                'lcs_r': round(lcs_r, 4), 'lci_r': round(lci_r, 4)
            },
            "datos_grafica": {
                'promedios': list(promedios_x), 'rangos': list(rangos_r)
            }
        }

    def crear_tabla_frecuencias(self, datos):
        valores_unicos, frecuencias = np.unique(datos, return_counts=True)
        n_total = len(datos); tabla = []; frecuencia_acumulada = 0
        for valor, freq in zip(valores_unicos, frecuencias):
            frecuencia_acumulada += int(freq)
            freq_relativa = float(freq) / n_total
            freq_relativa_acum = frecuencia_acumulada / n_total
            tabla.append({
                'valor': float(valor), 'frecuencia': int(freq), 'frecuencia_relativa': round(freq_relativa, 4),
                'frecuencia_acumulada': frecuencia_acumulada, 'frecuencia_relativa_acumulada': round(freq_relativa_acum, 4)
            })
        return tabla

    def generar_graficas(self, datos=None, tipo='desagrupado', tabla_frecuencias=None, subgrupos_data=None):
        graficas = {}
        def guardar_grafica():
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            img_string = base64.b64encode(buf.read()).decode()
            plt.close()
            return img_string

        plt.style.use('seaborn-v0_8-whitegrid')

        if tipo == 'desagrupado':
            # La lógica para datos desagrupados se mantiene igual
            plt.figure(figsize=(8, 5)); plt.hist(datos, bins='auto', alpha=0.75, color='skyblue', edgecolor='black'); plt.title('Histograma de Frecuencias'); plt.xlabel('Valores'); plt.ylabel('Frecuencia')
            media, mediana = np.mean(datos), np.median(datos)
            plt.axvline(media, color='red', linestyle='--', label=f'Media: {media:.2f}'); plt.axvline(mediana, color='green', linestyle='-.', label=f'Mediana: {mediana:.2f}'); plt.legend(); graficas['histograma'] = guardar_grafica()
            if media > mediana: graficas['sesgo_visual'] = "derecha (positivo)"
            elif media < mediana: graficas['sesgo_visual'] = "izquierda (negativo)"
            else: graficas['sesgo_visual'] = "simétrico"
            plt.figure(figsize=(6, 5)); plt.boxplot(datos, vert=True, patch_artist=True, boxprops={'facecolor': 'lightblue'}); plt.title('Diagrama de Caja y Bigotes'); plt.ylabel('Valores'); graficas['boxplot'] = guardar_grafica()
            if tabla_frecuencias:
                valores = [r['valor'] for r in tabla_frecuencias]; frec_rel = [r['frecuencia_relativa'] for r in tabla_frecuencias]; frec_acum = [r['frecuencia_acumulada'] for r in tabla_frecuencias]
                plt.figure(figsize=(8, 5)); plt.bar([str(v) for v in valores], frec_rel, color='mediumpurple', alpha=0.7); plt.title('Gráfica de Frecuencia Relativa'); plt.xlabel('Valores'); plt.ylabel('Frecuencia Relativa'); plt.xticks(rotation=45); graficas['frecuencia_relativa'] = guardar_grafica()
                plt.figure(figsize=(8, 5)); plt.bar([str(v) for v in valores], frec_acum, color='coral', alpha=0.7); plt.title('Gráfica de Frecuencia Acumulada'); plt.xlabel('Valores'); plt.ylabel('Frecuencia Acumulada'); plt.xticks(rotation=45); graficas['frecuencia_acumulada'] = guardar_grafica()
        
        elif tipo == 'agrupado' and subgrupos_data:
            metricas = subgrupos_data['metricas']
            datos_grafica = subgrupos_data['datos_grafica']
            fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
            muestras = np.arange(1, metricas['numero_muestras_k'] + 1)

            # Gráfica X-bar
            axs[0].plot(muestras, datos_grafica['promedios'], 'o-', color='darkcyan', label='Promedio de Muestra (X̄)')
            axs[0].axhline(metricas['gran_media_x_bar_bar'], color='blue', linestyle='--', label=f"LC = {metricas['gran_media_x_bar_bar']}")
            axs[0].axhline(metricas['lcs_x'], color='red', linestyle=':', label=f"LCS = {metricas['lcs_x']}")
            axs[0].axhline(metricas['lci_x'], color='red', linestyle=':', label=f"LCI = {metricas['lci_x']}")
            axs[0].set_title('Gráfica de Control X̄ (Promedios)', fontsize=14)
            axs[0].set_ylabel('Promedio de Muestra'); axs[0].legend(); axs[0].grid(True)

            # Gráfica R
            axs[1].plot(muestras, datos_grafica['rangos'], 'o-', color='orangered', label='Rango de Muestra (R)')
            axs[1].axhline(metricas['rango_promedio_r_bar'], color='blue', linestyle='--', label=f"LC = {metricas['rango_promedio_r_bar']}")
            axs[1].axhline(metricas['lcs_r'], color='red', linestyle=':', label=f"LCS = {metricas['lcs_r']}")
            axs[1].axhline(metricas['lci_r'], color='red', linestyle=':', label=f"LCI = {metricas['lci_r']}")
            axs[1].set_title('Gráfica de Control R (Rangos)', fontsize=14)
            axs[1].set_ylabel('Rango de Muestra'); axs[1].legend(); axs[1].grid(True)
            
            plt.xlabel('Número de Muestra', fontsize=12); plt.xticks(muestras)
            plt.tight_layout(pad=3.0)
            fig.suptitle('Gráficas de Control X-bar y R', fontsize=16, weight='bold')
            graficas['grafica_xr'] = guardar_grafica()
            
        return graficas

analizador = AnalisisEstadistico()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/procesar_datos', methods=['POST'])
def procesar_datos():
    data = request.get_json()
    try:
        analizador.es_muestral = data.get('es_muestral', True)
        es_agrupado = data.get('es_agrupado', False)

        if es_agrupado:
            # Nueva lógica para procesar tabla de subgrupos
            tabla_texto = data.get('datos', '')
            df = pd.read_csv(io.StringIO(tabla_texto), header=None, delim_whitespace=True, dtype=float)
            df.dropna(axis=1, how='all', inplace=True) # Eliminar columnas vacías si las hay
            
            subgrupos_data = analizador.calcular_analisis_subgrupos(df)
            graficas = analizador.generar_graficas(tipo='agrupado', subgrupos_data=subgrupos_data)
            resultado = {'tipo': 'agrupado', 'metricas_control': subgrupos_data['metricas'], 'graficas': graficas}
        else:
            datos = [float(d) for d in data.get('datos', [])]
            estadisticas = analizador.calcular_estadisticas_basicas(datos)
            tabla_frecuencias = analizador.crear_tabla_frecuencias(datos)
            graficas = analizador.generar_graficas(datos=datos, tipo='desagrupado', tabla_frecuencias=tabla_frecuencias)
            resultado = {'tipo': 'desagrupado', 'estadisticas': estadisticas, 'tabla_frecuencias': tabla_frecuencias, 'graficas': graficas}
        
        return jsonify({'status': 'success', 'resultado': resultado})
    except Exception as e:
        app.logger.error(f"Error en el procesamiento: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f"Ocurrió un error en el servidor: {e}"})

html_template = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Programa de Análisis Estadístico</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f0f2f5; padding: 20px; }
        .container { max-width: 1000px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); overflow: hidden; }
        .header { background: linear-gradient(135deg, #4c68d7 0%, #6e48aa 100%); color: white; padding: 25px 30px; text-align: center; }
        .header h1 { font-size: 2em; margin-bottom: 5px; }
        .content { padding: 30px; }
        .step { background: #fafafa; border-radius: 8px; padding: 25px; margin-bottom: 20px; border: 1px solid #e8e8e8; }
        .step h3 { color: #333; margin-bottom: 20px; font-size: 1.25em; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; font-weight: 600; color: #555; }
        input, textarea { width: 100%; padding: 12px; border: 1px solid #ccc; border-radius: 6px; font-size: 16px; transition: border-color 0.2s, box-shadow 0.2s; }
        textarea { font-family: 'Courier New', Courier, monospace; line-height: 1.5; }
        input:focus, textarea:focus { outline: none; border-color: #4c68d7; box-shadow: 0 0 0 2px rgba(76, 104, 215, 0.2); }
        .radio-group { display: flex; gap: 20px; margin-top: 10px; }
        .radio-item { display: flex; align-items: center; gap: 8px; }
        .radio-item input[type="radio"] { width: auto; }
        .btn { background: #4c68d7; color: white; border: none; padding: 12px 25px; border-radius: 6px; font-size: 16px; font-weight: 600; cursor: pointer; transition: background-color 0.2s; }
        .btn:hover { background: #3a50a8; }
        .btn-secondary { background-color: #6c757d; }
        .btn-secondary:hover { background-color: #5a6268; }
        .hidden { display: none; }
        .results { background: #fafafa; border-radius: 8px; padding: 25px; margin-top: 20px; border: 1px solid #e8e8e8;}
        .results-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 2px solid #4c68d7; padding-bottom: 10px; }
        .results h3 { color: #333; margin-top: 20px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .stat-item { background: white; padding: 15px; border-radius: 6px; border-left: 4px solid #4c68d7; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .stat-label { font-weight: 600; color: #666; font-size: 0.9em; text-transform: uppercase; }
        .stat-value { font-size: 1.4em; font-weight: 700; color: #333; margin-top: 5px; }
        .table-container { overflow-x: auto; margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #e0e0e0; }
        th { background: #f1f1f1; color: #333; font-weight: 600; }
        .chart-container { margin: 30px 0; text-align: center; }
        .chart-container img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
        .loading { display: flex; justify-content: center; align-items: center; height: 100px; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #4c68d7; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .alert-error { background: #fff0f0; border-left: 4px solid #e74c3c; color: #c0392b; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <header class="header"><h1>📊 Programa de Análisis Estadístico</h1><p>Análisis de datos descriptivos y de control de procesos</p></header>
        <div class="content">
            <div class="step" id="step1">
                <h3>1. Configuración del Análisis</h3>
                <div class="form-group"><label>¿Los datos son de una muestra o población? (Aplica a análisis Desagrupado)</label><div class="radio-group"><div class="radio-item"><input type="radio" id="muestral" name="tipo_datos" value="muestral" checked><label for="muestral">Muestral</label></div><div class="radio-item"><input type="radio" id="poblacional" name="tipo_datos" value="poblacional"><label for="poblacional">Poblacional</label></div></div></div>
                <div class="form-group"><label>Selecciona el tipo de análisis:</label><div class="radio-group"><div class="radio-item"><input type="radio" id="desagrupados" name="agrupamiento" value="desagrupados" checked><label for="desagrupados">Análisis Descriptivo (Desagrupado)</label></div><div class="radio-item"><input type="radio" id="agrupados" name="agrupamiento" value="agrupados"><label for="agrupados">Gráficas de Control X-R (Subgrupos)</label></div></div></div>
                <button class="btn" onclick="configurarAnalisis()">Continuar</button>
            </div>
            <div class="step hidden" id="step2">
                <h3>2. Ingreso de Datos</h3>
                <div id="datos-desagrupados"><div class="form-group"><label for="datos-input-desagrupado">Ingresa los datos separados por comas o espacios:</label><textarea id="datos-input-desagrupado" rows="6" placeholder="Ej: 12, 15 18.5, 20 22"></textarea></div></div>
                <div id="datos-agrupados" class="hidden"><div class="form-group"><label for="datos-input-agrupado">Pega tu tabla de datos aquí (filas=muestras, columnas=observaciones):</label><textarea id="datos-input-agrupado" rows="10" placeholder="4.85  4.861 4.86  4.858 4.862\n4.871 4.88  4.861 4.864 4.872\n..."></textarea></div></div>
                <button class="btn" onclick="procesarDatos()">Analizar Datos</button>
            </div>
            <div id="loading" class="loading hidden"><div class="spinner"></div></div>
            <div id="resultados" class="results hidden"><div id="contenido-resultados"></div></div>
        </div>
    </div>
    <script>
        function configurarAnalisis() {
            const esAgrupado = document.querySelector('input[name="agrupamiento"]:checked').value === 'agrupados';
            document.getElementById('datos-desagrupados').classList.toggle('hidden', esAgrupado);
            document.getElementById('datos-agrupados').classList.toggle('hidden', !esAgrupado);
            document.getElementById('step1').classList.add('hidden');
            document.getElementById('step2').classList.remove('hidden');
            document.getElementById('resultados').classList.add('hidden');
            document.getElementById('loading').classList.add('hidden');
        }

        async function procesarDatos() {
            const loadingDiv = document.getElementById('loading');
            const resultadosDiv = document.getElementById('resultados');
            const contenidoResultados = document.getElementById('contenido-resultados');
            
            loadingDiv.classList.remove('hidden');
            resultadosDiv.classList.add('hidden');
            contenidoResultados.innerHTML = '';
            
            try {
                const esMuestral = document.querySelector('input[name="tipo_datos"]:checked').value === 'muestral';
                const esAgrupado = document.querySelector('input[name="agrupamiento"]:checked').value === 'agrupados';
                let datosParaEnviar = { es_muestral: esMuestral, es_agrupado: esAgrupado };
                
                if (esAgrupado) {
                    const datosTexto = document.getElementById('datos-input-agrupado').value;
                    if (datosTexto.trim().length === 0) throw new Error('El campo de datos para subgrupos no puede estar vacío.');
                    datosParaEnviar.datos = datosTexto;
                } else {
                    const datosTexto = document.getElementById('datos-input-desagrupado').value;
                    const datosLimpios = datosTexto.trim().replace(/[,\\s]+/g, ',');
                    const datos = datosLimpios.split(',').map(d => parseFloat(d)).filter(d => !isNaN(d));
                    if (datos.length < 2) throw new Error('Se requieren al menos dos datos numéricos válidos.');
                    datosParaEnviar.datos = datos;
                }
                
                const response = await fetch('/procesar_datos', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(datosParaEnviar) });
                if (!response.ok) { const errorData = await response.json(); throw new Error(errorData.message || `Error del servidor: ${response.status}`); }
                const result = await response.json();
                if (result.status === 'success') { mostrarResultados(result.resultado); resultadosDiv.classList.remove('hidden'); } 
                else { throw new Error(result.message); }
                
            } catch (error) {
                contenidoResultados.innerHTML = `<div class="alert-error"><strong>Error:</strong> ${error.message}</div>`;
                resultadosDiv.classList.remove('hidden');
            } finally {
                loadingDiv.classList.add('hidden');
            }
        }

        function mostrarResultados(resultado) {
            let html = `<div class="results-header"><h3>📈 Resultados del Análisis</h3><button class="btn btn-secondary" onclick="nuevoAnalisis()">Nuevo Análisis</button></div>`;
            
            if (resultado.tipo === 'agrupado') {
                html += '<h3>Métricas del Gráfico de Control</h3><div class="stats-grid">';
                for (const [key, value] of Object.entries(resultado.metricas_control)) {
                    html += `<div class="stat-item"><div class="stat-label">${traducirLabel(key)}</div><div class="stat-value">${value}</div></div>`;
                }
                html += '</div>';
            } else {
                html += `<h3>Estadísticas Descriptivas</h3><div class="stats-grid">`;
                for (const [key, value] of Object.entries(resultado.estadisticas)) {
                    html += `<div class="stat-item"><div class="stat-label">${traducirLabel(key)}</div><div class="stat-value">${Array.isArray(value) ? value.join(', ') : value}</div></div>`;
                }
                html += '</div>';
                if (resultado.tabla_frecuencias) {
                    html += '<h3>📊 Tabla de Frecuencias</h3><div class="table-container"><table><thead><tr><th>Valor</th><th>Frecuencia</th><th>F. Relativa</th><th>F. Acumulada</th><th>F. Rel. Acum.</th></tr></thead><tbody>';
                    resultado.tabla_frecuencias.forEach(fila => { html += `<tr><td>${fila.valor}</td><td>${fila.frecuencia}</td><td>${fila.frecuencia_relativa}</td><td>${fila.frecuencia_acumulada}</td><td>${fila.frecuencia_relativa_acumulada}</td></tr>`; });
                    html += '</tbody></table></div>';
                }
            }
            
            if (resultado.graficas) {
                html += '<h3>🎨 Gráficas Visuales</h3>';
                const graficasOrden = resultado.tipo === 'agrupado' 
                    ? ['grafica_xr'] 
                    : ['histograma', 'frecuencia_acumulada', 'frecuencia_relativa', 'boxplot'];
                
                graficasOrden.forEach(key => {
                    if(resultado.graficas[key]) {
                        html += `<div class="chart-container"><h4>${traducirLabel(key)}</h4><img src="data:image/png;base64,${resultado.graficas[key]}" alt="${traducirLabel(key)}"></div>`;
                    }
                });
                if (resultado.graficas.sesgo_visual) {
                    html += `<p style="text-align:center; margin-top:15px;"><strong>Análisis de Sesgo Visual:</strong> La distribución parece tener un sesgo hacia la ${resultado.graficas.sesgo_visual}.</p>`;
                }
            }
            document.getElementById('contenido-resultados').innerHTML = html;
        }
        
        function nuevoAnalisis() {
            document.getElementById('resultados').classList.add('hidden');
            document.getElementById('step2').classList.add('hidden');
            document.getElementById('datos-input-desagrupado').value = '';
            document.getElementById('datos-input-agrupado').value = '';
            document.getElementById('step1').classList.remove('hidden');
        }

        function traducirLabel(key) {
            const traducciones = {
                // Desagrupado
                'media': 'Media', 'mediana': 'Mediana', 'moda': 'Moda', 'varianza': 'Varianza', 'desviacion_estandar': 'Desviación Estándar',
                'valor_minimo': 'Mínimo', 'valor_maximo': 'Máximo', 'rango': 'Rango',
                // Subgrupos (X-R)
                'numero_muestras_k': 'Nº de Muestras (k)', 'tamano_muestra_n': 'Tamaño de Muestra (n)',
                'gran_media_x_bar_bar': 'Gran Media (X̿)', 'rango_promedio_r_bar': 'Rango Promedio (R̄)',
                'lcs_x': 'LCS (Gráfica X̄)', 'lci_x': 'LCI (Gráfica X̄)',
                'lcs_r': 'LCS (Gráfica R)', 'lci_r': 'LCI (Gráfica R)',
                // Gráficas
                'histograma': 'Histograma', 'boxplot': 'Diagrama de Caja y Bigotes', 'frecuencia_relativa': 'Gráf. Frecuencia Relativa',
                'frecuencia_acumulada': 'Gráf. Frecuencia Acumulada', 'grafica_xr': 'Gráfica de Control X-R'
            };
            return traducciones[key] || key.replace(/_/g, ' ');
        }
    </script>
</body>
</html>
"""

# Crear directorio de templates si no existe
import os
if not os.path.exists('templates'):
    os.makedirs('templates')

# Guardar el template
with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(html_template.strip())

if __name__ == '__main__':
    app.run(debug=True)