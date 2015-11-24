## Cristian Gustavo Castro
## Modelacion y simulacion
## Proyecto 

import Image
import cv2
import cv
import numpy as np
import random
from numpy import arctan
from math import cos, sin, pi
from random import randint
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw



probabilidad_cruce = 0.8
probabilidad_mutar = 0.2
edges = []
generaciones = 15
rostro = "rostro3.jpg"

## Genera un nuevo cromosoma
def cromosoma():
	x = [str(randint(0,1)) for i in range(8)]
	y = [str(randint(0,1)) for i in range(8)]
	rx = [str(randint(0,1)) for i in range(8)]
	ry = [str(randint(0,1)) for i in range(8)]
	theta = [str(randint(0,1)) for i in range(7)]
   
	return ''.join(x+y+ry+ry+theta)


## Funcion de crossover entre dos cromosomas
def crossover(papa, mama): ## PMX
    
    tam = len(papa)
    a = randint(0,len(papa)-1)
    b = a
    while b == a:
        b = randint(0,len(papa)-1)
    
    
    if(b > a):
        cruce = papa[a:b]
        cruce2 = mama[a:b]
        p = papa[0:a]+cruce2+papa[b::]
        m = mama[0:a]+cruce+mama[b::]
    
    else:
        cruce = papa[b:a]
        cruce2 = mama[b:a]
        p = papa[0:b]+cruce2+papa[a::]
        m = mama[0:b]+cruce+mama[a::]
    
    return (p,m)

## Mutacion de un cromosoma
def mutacion(cromosoma):
    index = randint(0, len(cromosoma) - 1)
    mutated = list(cromosoma)
    if (mutated[index] == 0):
        mutated[index] = "1"
    else:
        mutated[index] = "0"

    return ''.join(mutated)


## Decodificacion de un cromosoma
def decodificar(cromosoma):

	x = cromosoma[:8]
	y = cromosoma[8:16]
	rx = cromosoma[16:24]
	ry = cromosoma[24:32]
	theta  = cromosoma[32:39]

	return (x,y,rx,ry,theta)

## Se establece poblacion inicial
def poblar(cantidad, poblacion):
	for i in range(cantidad):
		poblacion.append(cromosoma())

	return poblacion

## Se retorna la funcion de elipse generada en base a centro y eje mayor y menor
def elipse(x,y,cx,cy,a,b):

	z = (x-cx)**2 / a**2 + (y-cy)**2 / b**2
	return z

## Evaluacion de funcion de fitness para un cromosoma
def fitness(cromosoma, pixeles):

	(x,y,rx,ry,theta) = decodificar(cromosoma)

	cx = int(x,2)
	cy = int(y,2)
	a = int(rx,2)	
	b = int(ry,2)
	result = 0

	i = 0
	j = 0

	for row in pixeles:
		for col in row:
			if(col == 255):
				val = elipse(i,-j,cx,cy,a,b)
				if abs(2-val) < 1:
					result += 1
				j += 1
			i += 1	

	return result		

## Funcion de cruces para una poblacion completa

def cruces(poblacion, valores):

	tam = len(valores)
	nueva_pob = []
	for indice in range(tam):
		max_index = valores.index(max(valores))
		nueva_pob.append(poblacion[max_index])
		valores.remove(valores[max_index])


	tam_pob = len(nueva_pob)	

	for i in range(tam_pob):
		opcion_cruce = random.random()
		if opcion_cruce < probabilidad_cruce:
			if(i == tam_pob - 1):
			 	(nueva_pob[i], nueva_pob[0])  = crossover(nueva_pob[i], nueva_pob[0] )
			else:
				(nueva_pob[i], nueva_pob[i+1]) = crossover(nueva_pob[i], nueva_pob[i+1] )

	return nueva_pob	

## Funcion de mutaciones para una poblacion completa
def mutaciones(poblacion):

	for i in range(len(poblacion)):
		opcion_mutar = random.random()
		if opcion_mutar < probabilidad_mutar:
			poblacion[i] = mutacion(poblacion[i])

	return poblacion

## Funcion principal donde se realiza el algoritmo completo 
def simular():
	pixeles = procesamiento_imagen()
	valores = []
	poblacion_tmp = []
	maximos = []
	poblacion = []
	maximo_indice = 0
	iteracion = 0
	print "Inicia..."
	poblacion = poblar(20, poblacion)

	
	for generacion in range(generaciones):
		for individuo in poblacion:
			iteracion += 1
			valores.append(fitness(individuo, pixeles))
			if(iteracion == len(poblacion)):
				maximo_indice = valores.index(max(valores))
				maximos.append(poblacion[maximo_indice])
				poblacion_tmp = cruces(poblacion, valores)

		poblacion = poblacion_tmp
		poblacion = mutaciones(poblacion)

		valores = []
		iteracion = 0


	maxCromosomas = []
	for cromosoma in maximos:
		
		maxCromosomas.append(fitness(cromosoma, pixeles))


	ind_crom = maxCromosomas.index(max(maxCromosomas))

	(x,y,rx,ry,theta) = decodificar(maximos[ind_crom])

	dibujar_cuadrado(int(x,2), int(y,2), int(rx,2), int(ry,2))

## Inicio 
simular()





