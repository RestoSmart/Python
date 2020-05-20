# Importa la libreria JSON
import json
# Importa la libreria requests
import requests
# Importa la libreria pandas
import pandas as pd


class ETL:
    """
    Clase que contiene algunas de las funciones mas usadas dentro de los procesos de ETL
    """

    def JSONFileReader(self, rutaArchivo):
        """
        Funcion que permite leer un archivo JSON almacenado en una ruta especifica
        :param rutaArchivo: Ruta donde se almacena el archivo JSON
        :return df: Pandas DataFrame con la informacion contenida en el archivo JSON
        """
        # Se crea un diccionario a partir de la ruta del archivo
        my_dict = pd.read_json(rutaArchivo)

        # Convierte un dicionario en un data frame
        df = pd.DataFrame.from_dict(my_dict)

        # Se retrona el pandas DataFrame
        return df

    def columnsWithNull(self, pdDataFrame):
        """
        Funcion que identifica las columnas que tienen al menos un valor nulo
        :param pdDataFrame: Pandas DataFrame a analizar
        :return: result: Pandas Dataframe con la cantidad de registros nulos por columna
        """
        # Identificacion nulos
        null_columns = pdDataFrame.columns[pdDataFrame.isnull().any()]

        # Cantidad de registros nulos por columna
        result = pdDataFrame[null_columns].isnull().sum()

        return result

    def URLJSONFile(self, url, user=None, password=None):
        """
        Funcion que permite descargar informacion de un archivo JSON de una URL
        :param url: URL donde se realiza la peticion de información
        :param user: Usuario de autenticacion en caso de ser necesario
        :param password: Contraseña de autenticacion en caso de ser necesaria
        :return: En caso de que la respuesta sea atendida correctamente se retorna un pandas DataFrame
        """
        # Permite adquirir un archivo a partir e la url que se pasa por parametro
        json_response = requests.get(url, auth=(user, password))

        # Se obtiene el codigo de estado
        # 200 --> Es que el request fue atendido
        if json_response.status_code == 200:

            # Crear un diccionario a partir del texto que duelve la respuesta
            dictionary = json.loads(json_response.text)

            # Normaliza el JSON y lo comviertirlo en un DataFrame
            matriz = pd.DataFrame.from_dict(dictionary)

            return matriz
        else:
            print("Requerimiento no fue exitoso Status Code = ", json_response.status_code)
