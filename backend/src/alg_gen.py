import random

# Tamaño de la población y número de generaciones
TAMANO_POBLACION = 100
N_GENERACIONES = 50

# Probabilidades de cruce y mutación
PROBABILIDAD_CRUCE = 0.8
PROBABILIDAD_MUTACION = 0.2

def generar_carton_aleatorio():
	return [random.randint(1, 50) for _ in range(5)]  # Cartón con 5 números aleatorios entre 1 y 50

def generar_poblacion_inicial():
	return [generar_carton_aleatorio() for _ in range(TAMANO_POBLACION)]

def calcular_aptitud(carton):
	# Implementa aquí tu función de aptitud que evalúa la calidad del cartón
	# basándose en tu análisis previo y devuelve una puntuación mayor para cartones más probables de ganar.
	return random.randint(0, 100)  # En este ejemplo, se utiliza una función de aptitud aleatoria

def seleccionar_padres(poblacion):
	return sorted(poblacion, key=calcular_aptitud, reverse=True)[:TAMANO_POBLACION//2]

def cruzar_padres(padres):
	descendientes = []
	for i in range(0, len(padres), 2):
		padre1 = padres[i]
		padre2 = padres[i+1]
		punto_cruce = random.randint(1, len(padre1) - 1)
		descendiente1 = padre1[:punto_cruce] + padre2[punto_cruce:]
		descendiente2 = padre2[:punto_cruce] + padre1[punto_cruce:]
		descendientes.extend([descendiente1, descendiente2])
	return descendientes

def aplicar_mutacion(poblacion):
	for i in range(len(poblacion)):
		if random.random() < PROBABILIDAD_MUTACION:
			punto_mutacion = random.randint(0, len(poblacion[i]) - 1)
			poblacion[i][punto_mutacion] = random.randint(1, 50)

def encontrar_mejor_carton(poblacion):
	return max(poblacion, key=calcular_aptitud)

def algoritmo_genetico():
	poblacion = generar_poblacion_inicial()

	for generacion in range(N_GENERACIONES):
		for i in range(len(poblacion)):
			poblacion[i].aptitud = calcular_aptitud(poblacion[i])

		padres = seleccionar_padres(poblacion)
		descendientes = cruzar_padres(padres)
		poblacion = padres + descendientes

		aplicar_mutacion(poblacion)

	mejor_carton = encontrar_mejor_carton(poblacion)
	return mejor_carton

# Ejecutar el Algoritmo Genético y obtener el mejor cartón:
mejor_carton = algoritmo_genetico()
print("Mejor cartón:", mejor_carton)
