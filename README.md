# 14MBID_Code Project

El proyecto `14MBID_Code` se organiza en cinco carpetas principales, cada una con su respectiva función y contenido:

- **data**: Carpeta destinada al almacenamiento de todos los ficheros de datos. Contiene archivos en formatos `grib` y `gpkg`, siendo este último para las geometrías.

- **env**: En esta carpeta se sitúa el archivo `.cdsapirc` con la configuración de la API de Copernicus. Adicionalmente, se encuentra el archivo `environment.yml` con la configuración de paquetes y dependencias.

- **logs**: El objetivo de esta carpeta es mantener un registro de los resultados obtenidos para las distintas ejecuciones. Los resultados de cada ejecución se almacenan en un archivo JSON que detalla las métricas de rendimiento.

- **notebooks**: Aquí se ubica el archivo `14MBID_notebook.ipynb`. En este notebook se realizan las visualizaciones y gráficos que permiten evaluar las distintas aproximaciones.

- **src**: Contiene el archivo `reanalysis_cerra_module.py`. Este módulo permite, a través de parámetros configurables, ejecutar el flujo de carga definido en la sección anterior.

## Configuración del entorno

Para la configuración del entorno de desarrollo, se ha empleado Anaconda, una plataforma que permite la gestión de paquetes y entornos, ideal para trabajar con proyectos de Python, especialmente en el ámbito científico y de análisis de datos. Este trabajo utiliza Python versión 3.7.16. Las principales bibliotecas y dependencias requeridas para el proyecto incluyen `gdal`, `cfgrib`, `xarray`, `geopandas`, `numpy`, y `cdsapi`.

Para replicar este entorno en otra instalación de Anaconda, se debe ejecutar el siguiente comando:

```shell
conda env create -f enviroment.yml
