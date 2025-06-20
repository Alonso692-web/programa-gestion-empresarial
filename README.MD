# 📊 Programa de Análisis Estadístico con Flask

Una aplicación web completa para realizar análisis estadístico de datos, incluyendo medidas de tendencia central, dispersión, gráficas y análisis visual de distribuciones.

## 🎯 Características

### ✅ Funcionalidades Implementadas

**Configuración Inicial:**
- ✅ Selección entre datos muestrales o poblacionales
- ✅ Selección entre datos agrupados o desagrupados
- ✅ Interfaz intuitiva paso a paso

**Para Datos Desagrupados:**
- ✅ Medidas de tendencia central (media, mediana, moda)
- ✅ Medidas de dispersión (varianza, desviación estándar)
- ✅ Valores máximo, mínimo y rango
- ✅ Tabla de frecuencias completa
- ✅ Histograma con análisis de sesgo visual
- ✅ Diagrama de caja y bigotes

**Para Datos Agrupados:**
- ✅ Cálculo de parámetros: valor máximo, mínimo, rango, clases, amplitud
- ✅ Medidas de tendencia central ajustadas para datos agrupados
- ✅ Medidas de dispersión
- ✅ Curtosis y sesgo estadístico
- ✅ Histograma para datos agrupados
- ✅ Gráfica X-R (Promedios y Rangos)

**Análisis Visual:**
- ✅ Detección automática de sesgo (izquierda/derecha/simétrico)
- ✅ Gráficas interactivas y de alta calidad
- ✅ Interfaz moderna y responsive

## 📖 Cómo Usar

### Para Datos Desagrupados

1. **Configuración:**
   - Selecciona "Muestral" o "Poblacional"
   - Selecciona "Desagrupados"
   - Haz clic en "Continuar"

2. **Ingreso de Datos:**
   - Ingresa los datos separados por comas
   - Ejemplo: `12, 15, 18, 20, 22, 25, 28, 30`
   - Haz clic en "Analizar Datos"

3. **Resultados:**
   - Estadísticas descriptivas
   - Tabla de frecuencias
   - Histograma con análisis de sesgo
   - Diagrama de caja y bigotes

### Para Datos Agrupados

1. **Configuración:**
   - Selecciona "Muestral" o "Poblacional"
   - Selecciona "Agrupados"
   - Haz clic en "Continuar"

2. **Ingreso de Datos:**
   - Ingresa las clases en formato "min-max" (ej: `10-20`)
   - Ingresa la frecuencia para cada clase
   - Usa "+ Agregar Clase" para más clases
   - Haz clic en "Analizar Datos"

3. **Resultados:**
   - Parámetros de agrupación
   - Estadísticas ajustadas para datos agrupados
   - Curtosis y sesgo
   - Histograma agrupado
   - Gráfica X-R

## 📊 Ejemplos de Uso

### Ejemplo 1: Datos Desagrupados (Calificaciones)
```
Datos: 85, 92, 78, 88, 95, 82, 90, 87, 93, 89
Tipo: Muestral, Desagrupados
```

### Ejemplo 2: Datos Agrupados (Edades)
```
Clases y Frecuencias:
18-25: 15
26-35: 22
36-45: 18
46-55: 12
56-65: 8
Tipo: Poblacional, Agrupados
```

## 🎨 Características de la Interfaz

- **Diseño Moderno:** Interfaz limpia con gradientes y efectos visuales
- **Responsive:** Se adapta a diferentes tamaños de pantalla
- **Paso a Paso:** Guía intuitiva through el proceso de análisis
- **Gráficas de Alta Calidad:** Visualizaciones profesionales con matplotlib
- **Análisis Automático:** Interpretación automática de sesgo y distribución

### Análisis Estadístico Completo
- Cálculos precisos para muestras y poblaciones
- Fórmulas correctas según el tipo de datos
- Manejo de casos especiales (sin moda, etc.)

### Visualizaciones Profesionales
- Histogramas con líneas de media y mediana
- Detección visual automática de sesgo
- Gráficas X-R para control de calidad
- Diagramas de caja con análisis de outliers

### Interfaz Intuitiva
- Validación de datos en tiempo real
- Mensajes de error informativos
- Carga dinámica de resultados
- Diseño adaptable a móviles


**¡Disfruta analizando tus datos! 📊✨**