from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import base64
import io
#import pandas as pd
#import seaborn as sns
#from scipy import stats

#import json

app = Flask(__name__)

class AnalisisEstadistico:
    def __init__(self):
        self.es_muestral = True
    
    def calcular_estadisticas_basicas(self, datos):
        datos = np.array(datos)
        media = float(np.mean(datos))
        mediana = float(np.median(datos))
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
    
    def calcular_estadisticas_agrupadas(self, clases, frecuencias):
        puntos_medios = np.array([(float(c.split('-')[0]) + float(c.split('-')[1])) / 2 for c in clases])
        frecuencias = np.array(frecuencias)
        total_datos = np.sum(frecuencias)
        media = float(np.sum(puntos_medios * frecuencias) / total_datos)
        
        if self.es_muestral:
            varianza = float(np.sum(frecuencias * (puntos_medios - media)**2) / (total_datos - 1)) if total_datos > 1 else 0
        else:
            varianza = float(np.sum(frecuencias * (puntos_medios - media)**2) / total_datos)
        
        desviacion_std = float(np.sqrt(varianza))
        frecuencias_acum = np.cumsum(frecuencias)
        clase_mediana_idx = int(np.where(frecuencias_acum >= total_datos/2)[0][0])
        clase_modal_idx = int(np.argmax(frecuencias))
        clase_modal = clases[clase_modal_idx]
        
        # Sesgo y Curtosis
        momento3 = float(np.sum(frecuencias * (puntos_medios - media)**3) / total_datos)
        momento4 = float(np.sum(frecuencias * (puntos_medios - media)**4) / total_datos)
        sesgo = float(momento3 / (desviacion_std**3)) if desviacion_std > 0 else 0
        curtosis = float((momento4 / (desviacion_std**4)) - 3) if desviacion_std > 0 else 0
        
        return {
            'media': round(media, 4), 'mediana_aproximada': f"Clase {clases[clase_mediana_idx]}", 'moda': f"Clase modal: {clase_modal}",
            'varianza': round(varianza, 4), 'desviacion_estandar': round(desviacion_std, 4),
            'sesgo': round(sesgo, 4), 'curtosis': round(curtosis, 4)
        }
    
    def crear_tabla_frecuencias(self, datos):
        valores_unicos, frecuencias = np.unique(datos, return_counts=True)
        n_total = len(datos)
        tabla = []
        frecuencia_acumulada = 0
        for valor, freq in zip(valores_unicos, frecuencias):
            frecuencia_acumulada += int(freq)
            freq_relativa = float(freq) / n_total
            freq_relativa_acum = frecuencia_acumulada / n_total
            tabla.append({
                'valor': float(valor), 'frecuencia': int(freq), 'frecuencia_relativa': round(freq_relativa, 4),
                'frecuencia_acumulada': frecuencia_acumulada, 'frecuencia_relativa_acumulada': round(freq_relativa_acum, 4)
            })
        return tabla
    
    def generar_graficas(self, datos, tipo='desagrupado', clases=None, frecuencias=None, tabla_frecuencias=None):
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
        
        elif tipo == 'agrupado' and clases and frecuencias:
            # ### INICIO DE NUEVA LÓGICA PARA DATOS AGRUPADOS ###
            puntos_medios = np.array([(float(c.split('-')[0]) + float(c.split('-')[1])) / 2 for c in clases])
            
            # 1. Histograma
            plt.figure(figsize=(8, 5)); plt.bar(clases, frecuencias, alpha=0.75, color='skyblue', edgecolor='black'); plt.title('Histograma'); plt.xlabel('Clases'); plt.ylabel('Frecuencia'); plt.xticks(rotation=45); graficas['histograma'] = guardar_grafica()

            # 2. Diagrama de Caja y Bigotes (Simulado)
            simulated_data = np.repeat(puntos_medios, frecuencias)
            plt.figure(figsize=(6, 5)); plt.boxplot(simulated_data, vert=True, patch_artist=True, boxprops={'facecolor': 'lightgreen'}); plt.title('Diagrama de Caja (Simulado)'); plt.ylabel('Valores'); graficas['boxplot'] = guardar_grafica()

            # 3. Gráfica de Control X-R
            fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            rangos_clase = np.array([float(c.split('-')[1]) - float(c.split('-')[0]) for c in clases])
            
            # Gráfica X
            media_x = np.mean(puntos_medios); std_x = np.std(puntos_medios)
            axs[0].plot(clases, puntos_medios, 'o-', color='darkcyan', label='Promedio de Clase')
            axs[0].axhline(media_x, color='blue', linestyle='--', label='Línea Central (Media)')
            axs[0].axhline(media_x + 3 * std_x, color='red', linestyle=':', label='LCS')
            axs[0].axhline(media_x - 3 * std_x, color='red', linestyle=':', label='LCI')
            axs[0].set_title('Gráfica de Control X (Promedios)'); axs[0].set_ylabel('Valor Promedio'); axs[0].legend()
            
            # Gráfica R
            media_r = np.mean(rangos_clase)
            axs[1].plot(clases, rangos_clase, 'o-', color='orangered', label='Rango de Clase')
            axs[1].axhline(media_r, color='blue', linestyle='--', label='Línea Central (Rango Medio)')
            axs[1].set_title('Gráfica de Control R (Rangos)'); axs[1].set_ylabel('Rango'); axs[1].legend()
            plt.xlabel('Clases'); plt.xticks(rotation=45); plt.tight_layout(); graficas['grafica_xr'] = guardar_grafica()
            # ### FIN DE NUEVA LÓGICA ###
            
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
            clases = data.get('clases', [])
            frecuencias = [int(f) for f in data.get('frecuencias', [])]
            
            # ### INICIO DE NUEVOS CÁLCULOS PARA DATOS AGRUPADOS ###
            limites_inf = [float(c.split('-')[0]) for c in clases]
            limites_sup = [float(c.split('-')[1]) for c in clases]
            valor_min = min(limites_inf)
            valor_max = max(limites_sup)
            rango_total = valor_max - valor_min
            num_clases = len(clases)
            amplitud_promedio = rango_total / num_clases
            
            parametros = {
                'valor_maximo': round(valor_max, 4), 'valor_minimo': round(valor_min, 4),
                'rango': round(rango_total, 4), 'num_clases': num_clases,
                'amplitud': round(amplitud_promedio, 4)
            }
            # ### FIN DE NUEVOS CÁLCULOS ###

            estadisticas = analizador.calcular_estadisticas_agrupadas(clases, frecuencias)
            graficas = analizador.generar_graficas(None, tipo='agrupado', clases=clases, frecuencias=frecuencias)
            resultado = {'tipo': 'agrupado', 'estadisticas': estadisticas, 'graficas': graficas, 'parametros_agrupacion': parametros}
        else:
            datos = [float(d) for d in data.get('datos', [])]
            estadisticas = analizador.calcular_estadisticas_basicas(datos)
            tabla_frecuencias = analizador.crear_tabla_frecuencias(datos)
            graficas = analizador.generar_graficas(datos, tipo='desagrupado', tabla_frecuencias=tabla_frecuencias)
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
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }
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
        .clase-freq-item { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; }
        .add-clase-btn { background: #27ae60; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer; font-size: 14px; margin-top: 10px; }
        .remove-clase-btn { background: #e74c3c; color: white; border: none; padding: 5px 10px; border-radius: 50%; cursor: pointer; font-size: 12px; line-height: 1; }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>📊 Programa de Análisis Estadístico</h1>
            <p>Análisis de datos con medidas de tendencia central, dispersión y gráficas</p>
        </header>
        <div class="content">
            <div class="step" id="step1">
                <h3>1. Configuración del Análisis</h3>
                <div class="form-group"><label>¿Los datos son de una muestra o población?</label><div class="radio-group"><div class="radio-item"><input type="radio" id="muestral" name="tipo_datos" value="muestral" checked><label for="muestral">Muestral</label></div><div class="radio-item"><input type="radio" id="poblacional" name="tipo_datos" value="poblacional"><label for="poblacional">Poblacional</label></div></div></div>
                <div class="form-group"><label>¿Los datos están agrupados o desagrupados?</label><div class="radio-group"><div class="radio-item"><input type="radio" id="desagrupados" name="agrupamiento" value="desagrupados" checked><label for="desagrupados">Desagrupados</label></div><div class="radio-item"><input type="radio" id="agrupados" name="agrupamiento" value="agrupados"><label for="agrupados">Agrupados</label></div></div></div>
                <button class="btn" onclick="configurarAnalisis()">Continuar</button>
            </div>
            <div class="step hidden" id="step2">
                <h3>2. Ingreso de Datos</h3>
                <div id="datos-desagrupados"><div class="form-group"><label for="datos-input">Ingresa los datos separados por comas o espacios:</label><textarea id="datos-input" rows="4" placeholder="Ej: 12, 15 18.5, 20 22"></textarea></div></div>
                <div id="datos-agrupados" class="hidden"><div class="form-group"><label>Clases y Frecuencias:</label><div id="clases-container"><div class="clase-freq-item"><input type="text" placeholder="Clase (ej: 10-20)" class="clase-input"><input type="number" placeholder="Frecuencia" class="freq-input" min="1"><button type="button" class="remove-clase-btn" onclick="removerClase(this)">×</button></div></div><button type="button" class="add-clase-btn" onclick="agregarClase()">+ Agregar Clase</button></div></div>
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
                    const clases = Array.from(document.querySelectorAll('.clase-input')).map(i => i.value.trim()).filter(Boolean);
                    const frecuencias = Array.from(document.querySelectorAll('.freq-input')).map(i => i.value).filter(Boolean);
                    if (clases.length === 0 || clases.length !== frecuencias.length) { throw new Error('Asegúrate de que cada clase tenga una frecuencia válida.'); }
                    datosParaEnviar.clases = clases;
                    datosParaEnviar.frecuencias = frecuencias;
                } else {
                    const datosTexto = document.getElementById('datos-input').value;
                    const datosLimpios = datosTexto.trim().replace(/[,\\s]+/g, ',');
                    const datos = datosLimpios.split(',').map(d => parseFloat(d)).filter(d => !isNaN(d));
                    if (datos.length < 2) { throw new Error('Se requieren al menos dos datos numéricos válidos.'); }
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
            
            // ### INICIO DE CAMBIOS EN LA VISUALIZACIÓN ###
            // Parámetros de Agrupación (si existen)
            if (resultado.parametros_agrupacion) {
                html += '<h3>Parámetros de la Tabla</h3><div class="stats-grid">';
                for (const [key, value] of Object.entries(resultado.parametros_agrupacion)) {
                    html += `<div class="stat-item"><div class="stat-label">${traducirLabel(key)}</div><div class="stat-value">${value}</div></div>`;
                }
                html += '</div>';
            }

            // Estadísticas Principales
            html += `<h3>${resultado.tipo === 'agrupado' ? 'Medidas Descriptivas (Agrupados)' : 'Estadísticas Descriptivas'}</h3><div class="stats-grid">`;
            for (const [key, value] of Object.entries(resultado.estadisticas)) {
                html += `<div class="stat-item"><div class="stat-label">${traducirLabel(key)}</div><div class="stat-value">${Array.isArray(value) ? value.join(', ') : value}</div></div>`;
            }
            html += '</div>';
            
            // Tabla de Frecuencias (solo para desagrupados)
            if (resultado.tabla_frecuencias) {
                html += '<h3>📊 Tabla de Frecuencias</h3><div class="table-container"><table><thead><tr><th>Valor</th><th>Frecuencia</th><th>F. Relativa</th><th>F. Acumulada</th><th>F. Rel. Acum.</th></tr></thead><tbody>';
                resultado.tabla_frecuencias.forEach(fila => { html += `<tr><td>${fila.valor}</td><td>${fila.frecuencia}</td><td>${fila.frecuencia_relativa}</td><td>${fila.frecuencia_acumulada}</td><td>${fila.frecuencia_relativa_acumulada}</td></tr>`; });
                html += '</tbody></table></div>';
            }
            
            // Gráficas
            if (resultado.graficas) {
                html += '<h3>🎨 Gráficas Visuales</h3>';
                const graficasOrden = resultado.tipo === 'agrupado' 
                    ? ['histograma', 'boxplot', 'grafica_xr'] 
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
            // ### FIN DE CAMBIOS ###
            
            document.getElementById('contenido-resultados').innerHTML = html;
        }
        
        function nuevoAnalisis() {
            document.getElementById('resultados').classList.add('hidden');
            document.getElementById('step2').classList.add('hidden');
            document.getElementById('datos-input').value = '';
            const clasesContainer = document.getElementById('clases-container');
            while (clasesContainer.children.length > 1) { clasesContainer.removeChild(clasesContainer.lastChild); }
            if (clasesContainer.children.length > 0) { clasesContainer.querySelector('.clase-input').value = ''; clasesContainer.querySelector('.freq-input').value = ''; }
            document.getElementById('step1').classList.remove('hidden');
        }

        function agregarClase() {
            const container = document.getElementById('clases-container');
            const div = document.createElement('div');
            div.className = 'clase-freq-item';
            div.innerHTML = `<input type="text" placeholder="Clase (ej: 10-20)" class="clase-input"><input type="number" placeholder="Frecuencia" class="freq-input" min="1"><button type="button" class="remove-clase-btn" onclick="removerClase(this)">×</button>`;
            container.appendChild(div);
        }

        function removerClase(btn) {
            if (btn.parentElement.parentElement.children.length > 1) btn.parentElement.remove();
        }

        function traducirLabel(key) {
            const traducciones = {
                'media': 'Media', 'mediana': 'Mediana', 'moda': 'Moda', 'varianza': 'Varianza', 'desviacion_estandar': 'Desviación Estándar',
                'valor_minimo': 'Mínimo', 'valor_maximo': 'Máximo', 'rango': 'Rango', 'sesgo': 'Coef. de Sesgo', 'curtosis': 'Coef. de Curtosis',
                'mediana_aproximada': 'Clase de la Mediana', 'num_clases': 'Nº de Clases', 'amplitud': 'Amplitud de Clase',
                'histograma': 'Histograma', 'boxplot': 'Diagrama de Caja y Bigotes', 'frecuencia_relativa': 'Gráf. Frecuencia Relativa',
                'frecuencia_acumulada': 'Gráf. Frecuencia Acumulada', 'grafica_xr': 'Gráfica de Control X-R'
            };
            return traducciones[key] || key.replace('_', ' ');
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