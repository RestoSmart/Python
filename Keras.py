# Importa la libreria Sequiential de Keras Tensorflow
from tensorflow.keras.models import Sequential
# Importa la libreria del optimizador Adam de Keras Tensorflow
from tensorflow.keras.optimizers import Adam
# Importa la libreria de capa Dense de Keras Tensorflow
from tensorflow.keras.layers import Dense
# Se importa la libreria de Pandas
import pandas as pd


class useKeras:
    """
    Clase que contiene algunas de las funciones para ejecutar modelos de Machine Learning con Keras
    """

    def modeloKeras(self, cantCapas, activadores, Xtrain, Ytrain, imputVaraiblesInicial, loss, restoPorCapa=0,
                    optimizador=Adam(lr=0.05), metricas=['accuracy'], epochsModel=200):
        """
        Funcion que crea un modelo de Tensorflow Keras a partir de los parametros recibidos
        :param cantCapas: Cantidad de capas que se quieren para el modelo
        :param activadores: Arreglo de textos con los activadores de cada capa
        :param Xtrain: Pandas DataFrame con las Variables predictoras
        :param Ytrain: Pandas DataFrame con la variable a predecir
        :param imputVaraiblesInicial: Cantidad de variavles que se van a tomar para la primera capa de la red neuronal
        :param loss: Variable de medida a minimizar durante el proceso de entrenamiento
        :param restoPorCapa: Cantidad de varaibles que se van a restar por cada una de las capas, por defecto no se
        resta ninguna
        :param optimizador: Optimizador que se va a utlizar para la evaluacion de la red neuronal, por defecto Adam con
        un learning rate de 0.05
        :param metricas: Arreglo con las metricas que se van a tener en cuenta para la evaluacion del modelo, por
        defecto el accuracy
        :param epochsModel: Cantidad de veces que se evalua el modelo
        :return: El modelo de keras compilado
        """
        # Cantidad de varaibles
        shapeX = Xtrain.shape[1]

        # Se inicializa el modelo
        modelKeras = Sequential()

        # Se inicializa las variables restantes en el imput de variables inicial
        variablesRestantes = imputVaraiblesInicial

        # Arreglo para insercion de capas intermedias
        for i in range(cantCapas):
            # Si es la primera capa
            if i == 0:
                modelKeras.add(Dense(imputVaraiblesInicial, input_shape=(shapeX,), activation=activadores[i]))
            # Si es una capa intermedia
            elif i < (cantCapas - 1):
                modelKeras.add(Dense((variablesRestantes - restoPorCapa), activation=activadores[i]))
                # Se actualiza el total de variables restantes
                variablesRestantes -= restoPorCapa
            # El resto de los casos
            else:
                modelKeras.add(Dense(1, activation=activadores[i]))

        # Se complia el modelo con el optimizador Adam (Learning Rate de 0.05) y metrica del Accuracy
        modelKeras.compile(optimizer=optimizador, loss=loss, metrics=metricas)

        # Se imprime en consola un resumen del contexto del modelo
        print(modelKeras.summary())

        # Se entrena el modelo con los datos de prueba. Con 200 iteraciones sobre y sin print en pantalla
        modelKeras.fit(Xtrain, Ytrain, epochs=epochsModel, verbose=0)

        # Se retorna el modelo
        return modelKeras

    def evaluacionKeras(self, modelKeras, Xtest, Ytest, metricas=['accuracy']):
        """
        Funcion que permite evaluar un modelo de Keras
        :param modelKeras: Modelo de Keras construido previamente
        :param Xtest: Variables de entrenamiento
        :param Ytest: Varaible a predecir
        :param metricas: Metricas que se toman de validación, por defecto unicamente el accuracy
        :return: Pandas DataFrame con la informacion resultante de evaluar el modelo
        """
        # Se evalua el modelo de Keras por parametro
        modelResult = modelKeras.evaluate(Xtest, Ytest, verbose=0)

        # Arreglo que guardara la informacion de las metricas evaluadas
        arregloDF = []

        for i in range(len(modelResult)):
            # Se valida la primera medida que siempre es loss
            if i == 0:
                arregloDF.append(["loss", modelResult[i]])
            # En caso contrario se agrega la metrica de valoracion del modelos
            else:
                arregloDF.append([metricas[i - 1], modelResult[i]])

        return pd.DataFrame(arregloDF, columns=["medida", "resultado"])
