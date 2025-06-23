# 📊 Programa de Análisis Estadístico

Una aplicación web desarrollada en Flask para realizar análisis estadísticos descriptivos y gráficas de control de procesos de manera intuitiva y profesional.

## ✨ Características Principales

### 🔢 Datos Individuales
- **Estadísticas básicas**: Media, mediana, moda, varianza, desviación estándar
- **Medidas de forma**: Coeficiente de sesgo, curtosis con interpretaciones automáticas
- **Rango y valores extremos**: Mínimo, máximo y rango de los datos
- **Tabla de frecuencias**: Frecuencia absoluta, relativa, acumulada y relativa acumulada

### 📈 Datos Agrupados
- **Análisis de subgrupos**: Soporte para tamaños de muestra de 2 a 20 observaciones
- **Constantes de control**: Utiliza constantes estándar de ingeniería (A2, D3, D4)
- **Límites de control**: Cálculo automático de límites superior e inferior
- **Gráficas duales**: Visualización simultánea de gráficas X̄ y R

### 🎨 Visualizaciones Interactivas
- **Histogramas** con líneas de referencia (media y mediana)
- **Diagramas de caja** para identificar valores atípicos
- **Gráficas de frecuencia** (relativa y acumulada)
- **Gráficas de control** profesionales con límites claramente marcados

## 🛠️ Tecnologías Utilizadas

- **Backend**: Flask (Python)
- **Análisis**: NumPy, Pandas, SciPy
- **Visualización**: Matplotlib
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Diseño**: Interfaz responsiva con gradientes modernos

## 📋 Requisitos del Sistema

```
Python 3.7+
Flask
NumPy
Pandas
Matplotlib
SciPy
```

## 🚀 Instalación y Configuración

1. **Clona el repositorio**:
```bash
git clone https://github.com/Alonso692-web/programa-gestion-empresarial
cd programa-analisis-estadistico
```

2. **Instala las dependencias**:
```bash
pip install flask numpy pandas matplotlib scipy
```

3. **Ejecuta la aplicación**:
```bash
python app.py
```

4. **Accede a la aplicación**:
Abre tu navegador y visita `http://localhost:5000`

## 📖 Guía de Uso

### Análisis Descriptivo (Datos Individuales)

1. **Configuración**: Selecciona "Muestral" o "Poblacional" según tu tipo de datos
2. **Tipo de análisis**: Elige "Análisis Descriptivo (Individual)"
3. **Ingreso de datos**: Introduce los valores separados por comas o espacios
   ```
   Ejemplo: 12, 15, 18.5, 20, 22, 19, 16, 14
   ```
4. **Resultados**: Obtén estadísticas completas y visualizaciones automáticas

### Gráficas de Control X-R (Subgrupos)

1. **Configuración**: Selecciona el tipo de datos
2. **Tipo de análisis**: Elige "Gráficas de Control X-R (Subgrupos)"
3. **Formato de datos**: Pega tu tabla donde:
   - Cada fila = una muestra/subgrupo
   - Cada columna = una observación
   ```
   Ejemplo:
   4.85  4.861 4.86  4.858 4.862
   4.871 4.88  4.861 4.864 4.872
   4.869 4.863 4.855 4.870 4.865
   ```
4. **Análisis**: Obtén métricas de control y gráficas profesionales

## 📊 Interpretación de Resultados

### Estadísticas Descriptivas

- **Sesgo**: 
  - Positivo: Distribución asimétrica hacia la derecha
  - Negativo: Distribución asimétrica hacia la izquierda
  - Cercano a 0: Distribución aproximadamente simétrica

- **Curtosis**:
  - Leptocúrtica (>0.2): Distribución más apuntada que la normal
  - Mesocúrtica (≈0): Distribución similar a la normal
  - Platicúrtica (<-0.2): Distribución más aplanada que la normal

### Gráficas de Control

- **LCS/LCI**: Límites de control superior e inferior
- **LC**: Línea central (promedio del proceso)
- **Puntos fuera de límites**: Indican posible descontrol del proceso

## 🎯 Casos de Uso

- **Control de calidad industrial**
- **Análisis de procesos de manufactura**
- **Investigación académica y educativa**
- **Análisis de datos en ingeniería**
- **Estudios de capacidad de procesos**

## 📁 Estructura del Proyecto

```
ANALISIS-ESTADISTICO/
│
├── templates/            # Plantillas HTML
│   └── index.html       # Interfaz de usuario principal
├── venv_project/        # Entorno virtual de Python
├── app.py               # Aplicación principal Flask
├── Procfile             # Configuración para despliegue
├── README.md            # Documentación del proyecto
├── requirements.txt     # Dependencias del proyecto
└── runtime.txt          # Versión de Python para despliegue
```

## 🔧 Características Técnicas

### Constantes de Control Soportadas
- Tamaños de muestra: n = 2 hasta n = 20
- Constantes A2, D3, D4 según estándares de ingeniería
- Cálculos basados en métodos estadísticos reconocidos

### Algoritmos Implementados
- **Sesgo de Pearson**: Coeficientes basados en moda y mediana
- **Curtosis de Fisher**: Medida de apuntamiento ajustada
- **Estadísticas robustas**: Manejo de valores atípicos
- **Validación de datos**: Filtros automáticos para datos inválidos

## 👥 Desarrolladores

- **Raul Cardoso Acevedo**
- **Alonso Domínguez López**

## 📝 Notas Importantes

- Los datos deben ser numéricos válidos
- Para gráficas de control, se requieren al menos 2 subgrupos
- El tamaño de muestra debe estar entre 2 y 20 observaciones
- La aplicación maneja automáticamente valores faltantes

## 📄 Licencia

La de dios padres🙏

---

*Para soporte técnico o sugerencias, contacta a los desarrolladores.*