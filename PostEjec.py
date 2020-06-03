class PostEjec:
    """
    Clase que contiene algunas de las funciones para validacion de los resultados de algunos modelos de Machine Learning
    """

    # Importacion de seaborn
    import seaborn as sns
    # Importacion de pyplot
    import matplotlib.pyplot as plt
    # Importacion de numpy
    import numpy as np
    # Importacion de pandas
    import pandas as pd
    # Importacion de export_graphviz
    from sklearn.tree import export_graphviz
    # Importacion de pydot
    import pydot

    def __init__(self):
        pass

    def grafMatrizConfusion(self, matriz, labels, titulo="Matriz de confusión", tamano=(5, 5), rutaImg=None):
        """
        Funcion que permite crear un grafico de la matriz de confusión
        :param matriz: Resultado de la matriz de confusion, se puede obtener con el metodo
        sklearn.metrics.confusion_matrix
        :param labels: Valores del las etiquetas de los ejes
        :param titulo: Titulo del grafico, default "Matriz de confusión"
        :param tamano: Tamaño del grafico, default 5 x 5
        :param rutaImg: Ruta donde se guardara la imagen, default None
        """
        # Se crea un graficon con tamaño 10 de ancho y 2.5 de alto
        fig, ax = plt.subplots(figsize=tamano)

        # Se muestra el grafico completo
        fig.tight_layout()

        # Se adiciona el mapa de calor al grafico
        ax = sns.heatmap(matriz, annot=True, cmap="Blues", annot_kws={"size": 14}, fmt='g', cbar=False)

        # Se agrega el titulo
        plt.title(titulo + "\n", fontsize=9, fontweight='bold')

        # Se agrega el titulo del eje y
        plt.ylabel('Predicted label \n', fontsize=13, fontweight='bold')

        # Se agrega el titulo del eje x
        plt.xlabel('Actual label \n', fontsize=13, fontweight='bold')

        # Se agrega la etiqueta de datos del eje x
        plt.xticks(np.arange(len(labels)) + 0.5, labels, fontsize=14)

        # Se agrega la etiqueta de datos del eje y
        plt.yticks(np.arange(len(labels)) + 0.5, labels, fontsize=14, rotation=0)

        # Se posiciona el eje x arriba
        ax.xaxis.tick_top()

        # Se posiciona los labels arriba
        ax.xaxis.set_label_position('top')

        if rutaImg is not None:
            # Se salva el grafico como una imagen
            plt.savefig(rutaImg, bbox_inches="tight")
            print("Se guardo la imagen en la siguiente ruta: " + rutaImg)

        # Se muestra el grafico
        plt.show()

    def graficoImportancia(self, featureImportance, tamano=(5, 5), rutaImg=None):
        """
        Funcion que muestra la importacion de cada variable dentro del modelo en un grafico de barras
        :param featureImportance: Pandas DataFrame con la informacion de la importancia de las varaibles
        :param tamano: Tamaño del grafico, default 5 x 5
        :param rutaImg: Ruta donde se guardara la imagen, default None
        """
        if "valor" not in featureImportance.columns and "variable" not in featureImportance.columns:
            print("El DataFrame que se debe pasar debe contener las columnas variable y valor")

        else:
            # Se crea un graficon con tamaño 10 de ancho y 2.5 de alto
            fig, ax = plt.subplots(figsize=tamano)

            # Se muestra el grafico completo
            fig.tight_layout()

            df = featureImportance.sort_values(by="valor", ascending=False)

            posicion = np.arange(len(df["variable"]))

            ax = plt.bar(posicion, df["valor"])

            plt.xticks(posicion, df["variable"], rotation='vertical')

            # Se posiciona los labels arriba
            ax.xaxis.set_label_position('bottom')

            if rutaImg is not None:
                # Se salva el grafico como una imagen
                plt.savefig(rutaImg, bbox_inches="tight")
                print("Se guardo la imagen en la siguiente ruta: " + rutaImg)

            plt.show()

    def graficoArbolDesicion(self, desicisionTree, variables):
        """
        Funcion que permite crear un archivo de imagen con un arbol de desicion
        :param variables: Nombre de las variables
        :param desicisionTree: SKLearn DecisionTreeRegressor
        """
        export_graphviz(desicisionTree, out_file='tree.dot', feature_names=list(variables), rounded=True, proportion=False,
                        precision=2, filled=True)

        # Use dot file to create a graph
        (graph,) = pydot.graph_from_dot_file('tree.dot')
        # Write graph to a png file
        graph.write_png('tree.png')

        # Display in jupyter notebook
        from IPython.display import Image
        Image(filename='tree.png')
