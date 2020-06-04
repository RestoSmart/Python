# Importacion del modelo de KMeans
from sklearn.cluster import KMeans
# Importa la libreria MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
# Importa la libreria StandardScaler
from sklearn.preprocessing import StandardScaler


class Preprocessing:
    """
    Clase que contiene algunas de las funciones mas usadas dentro de los procesos de aprendizaje automatico utilizando
    la libreria de Scikit learn
    """

    def __init__(self):
        pass

    def numberClusterElbowKMeans(self, dfModel, maxClusters=10):
        """
        Funcion que permite realizar el algoritmo de ElBow con el fin de determinar el numero de clusters adecuado en
        un modelo KMeans
        :param maxClusters: Numero maximo de clusters a probrar, valor default 10
        :param dfModel: Pandas DataFrame sobre el cual se ejecutara el algoritmo de KMeans
        :return: nclusters: Arreglo con la cantidad de clusters
        :return: inertias: Inercia del modelo KMeans dependiendo la cantidad de grupos
        """
        # Se crea un arreglo desde 1 hasta la cantidad maxima de clusters
        nclusters = range(1, maxClusters)

        # Se crea un arreglo vacio donde se almacenara la inercia de cada modelo
        inertias = []

        for i in nclusters:
            # Se define el modelo de KMeans con la cantidad de clusters definidos
            kMeansModel = KMeans(n_clusters=i)

            # Se realiza el entrenamiento del model KMeans
            kMeansModel.fit(dfModel)

            # Se actualiza el valor de la inercia para la cantidad de clusters definidoss
            inertias.append(kMeansModel.inertia_)

        return nclusters, inertias

    def minMaxScaleColumns(self, dataFrameColumnasNumericas):
        """
        Funcion que permite ejecutar la transformacion de los datos escalando cada una de las variables entre sus
        valores maximos y minimos
        :param dataFrameColumnasNumericas: Pandas DataFrame con los atributos que se van a escalar
        :return: Pandas DataFrame ya escalado
        """
        # Se inicializa el MinMaxScaler
        minMax = MinMaxScaler()

        # Se realiza el ajuste a partir de los maximos y minimos para cada una de las variables numericas
        minMax.fit(dataFrameColumnasNumericas)

        # Se retorna el DataFrame ajustado con la transformacion de min max
        return minMax.transform(dataFrameColumnasNumericas)

    def standardScalerColumns(self, dataFrameColumnasNumericas):
        """
        Funcion que permite la ejecucion de la transformacion de los datos escalando cada una de las variables con su
        media y desviación estandar
        :param dataFrameColumnasNumericas: Pandas DataFrame con los atributos que se van a escalar
        :return: Pandas DataFrame ya escalado
        """
        # Se inicializa el MinMaxScaler
        standardScaler = StandardScaler()

        # Se realiza el ajuste a partir de la media y la desviacion estandar para cada una de las variables numericas
        standardScaler.fit(dataFrameColumnasNumericas)

        # Se retorna el DataFrame ajustado con la transformacion de media y desviacion estandar
        return standardScaler.transform(dataFrameColumnasNumericas)

    def labelEncoderTexto(self, dataFrameColumnasTexto):
        """
        Funcion que asigna un valor numerico para cada una de las atributos texto de un DataFrame
        :param dataFrameColumnasTexto: Atributos de tipo texto que se les realizara la transformacion
        :return: diccionarioEncoder: El diccionario que referencia cada uno de los valores textos con los valores
        numericos
        :return: dfRetorno: Pandas DataFrame con las variables ya asignadas a numeros numericos
        """
        # Se inicializa el LabelEncoder
        encoder = LabelEncoder()

        # Se crea un diccionario para las transformaciones
        diccionarioEncoder = {}

        dfRetorno = dataFrameColumnasTexto.copy()

        for i in dfRetorno.columns:
            # Se realiza la codificacion
            encoder.fit(dfRetorno[i])

            # Obtencion de la codificacion
            diccionarioEncoder.update({i: encoder.classes_})

            # Se realiza la actulizacion del campo
            dfRetorno[i] = encoder.transform(dfRetorno[i])

        return diccionarioEncoder, dfRetorno
