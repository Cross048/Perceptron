from neurona import Neurona


class RedNeuronal:
    def __init__(self, estructura):
        # Estructura es una lista que indica cuántas neuronas tiene cada capa
        self.capas = []
        for i in range(1, len(estructura)):
            capa = [Neurona(estructura[i - 1]) for _ in range(estructura[i])]
            self.capas.append(capa)

    def pasada_adelante(self, entradas):
        activaciones = entradas
        for capa in self.capas:
            nueva_activacion = [neurona.salida(activaciones) for neurona in capa]
            activaciones = nueva_activacion
        return activaciones

    def retropropagacion(self, entradas, resultado_esperado, tasa_aprendizaje):
        # Paso 1: Pasada hacia adelante
        activaciones = [entradas]
        for capa in self.capas:
            activacion_capa = [neurona.salida(activaciones[-1]) for neurona in capa]
            activaciones.append(activacion_capa)

        # Mostrar activaciones para depuración
        print(f"Activaciones: {activaciones}")

        # Paso 2: Calcular errores en la capa de salida
        errores = [None] * len(self.capas)
        errores[-1] = [
            (resultado_esperado[i] - activaciones[-1][i]) * self.capas[-1][i].derivada_sigmoide(activaciones[-1][i])
            for i in range(len(activaciones[-1]))
        ]

        # Mostrar errores para depuración
        print(f"Errores de salida: {errores[-1]}")

        # Paso 3: Propagación de errores hacia atrás
        for i in range(len(self.capas) - 2, -1, -1):
            errores[i] = [
                sum(errores[i + 1][j] * self.capas[i + 1][j].pesos[k] for j in range(len(self.capas[i + 1]))) *
                self.capas[i][k].derivada_sigmoide(activaciones[i + 1][k])
                for k in range(len(self.capas[i]))
            ]

            # Mostrar errores de la capa oculta para depuración
            print(f"Errores en la capa {i}: {errores[i]}")

        # Paso 4: Actualización de pesos y sesgos
        for i in range(len(self.capas)):
            for j, neurona in enumerate(self.capas[i]):
                neurona.ajustar_pesos(activaciones[i], errores[i][j], tasa_aprendizaje)

    def entrenar(self, entradas, resultados_esperados, tasa_aprendizaje, ciclos):
        for ciclo in range(ciclos):
            for entrada, resultado in zip(entradas, resultados_esperados):
                self.retropropagacion(entrada, [resultado], tasa_aprendizaje)
