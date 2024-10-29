import math
import random


class Neurona:
    def __init__(self, num_entradas):
        # Inicializa los pesos y el sesgo aleatoriamente para cada neurona
        self.pesos = [random.random() for _ in range(num_entradas)]
        self.sesgo = random.random()

    def salida(self, entradas):
        # Calcula la salida de la neurona usando la función de activación sigmoide
        salida_suma = sum(p * e for p, e in zip(self.pesos, entradas)) + self.sesgo
        return self.funcion_sigmoide(salida_suma)

    def funcion_sigmoide(self, salida):
        return 1 / (1 + math.exp(-salida))

    def derivada_sigmoide(self, salida):
        return salida * (1 - salida)

    def ajustar_pesos(self, entradas, error, tasa_aprendizaje):
        # Ajusta los pesos y el sesgo usando la regla delta
        for i in range(len(self.pesos)):
            self.pesos[i] += tasa_aprendizaje * error * entradas[i]
        self.sesgo += tasa_aprendizaje * error
