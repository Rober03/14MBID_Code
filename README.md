# 14MBID_Code

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
```

Por defecto, al utilizar este archivo, se creará un entorno con el nombre `tfm`. Sin embargo, si se desea asignar un nombre diferente al entorno, es posible hacerlo añadiendo la opción `--name` seguido del nombre deseado:

```shell
conda env create -f enviroment.yml --name nombre_entorno
```

## Configuración de la API de Copernicus

1. **Registro en Copernicus**: Dirígete y regístrate en la página oficial de Copernicus: [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)

2. **Acceso a la API Key**: Una vez que hayas iniciado sesión, selecciona el nombre de usuario situado en la esquina superior izquierda. En el menú desplegado, encontrarás la información referente a la API Key, abarcando tanto el UID como la API Key propiamente dicha.

3. **Configuración del archivo `.cdsapirc`**: Con los datos obtenidos, crea un archivo con el siguiente contenido:

```makefile
url: https://cds.climate.copernicus.eu/api/v2
key: UID:API_KEY
```
4. Asegúrate de que este archivo tenga por nombre .cdsapirc. Guarda el archivo .cdsapirc en la ruta ~/.cdsapirc para que pueda ser detectado y utilizado correctamente.

Si surgen dificultades al generar la API Key, es posible recurrir al archivo .cdsapirc proporcionado en la carpeta env del proyecto. El archivo contiene la información de la API Key asociada a mi perfil.

