
#install.packages('bnlearn')
require(e1071)
require(RWeka)
####################################################################################
####################################################################################
####################################################################################
######### NUESTRO #########
####################################################################################
####################################################################################
####################################################################################

url.data.set <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/00329/messidor_features.arff'
data.raw <- read.csv(url.data.set, header=FALSE, comment.char = "@")
df <- data.frame(data.raw)
colnames(df) <- c(
  "q",      #  0 Numero binario, relacionado con la calidad de la imagen. 0 = Mala calidad; 1 = Buena calidad.(remover)
  "ps",     #  1 Numero binario resultante de un pre-escaneo, que indica si la imagen presenta animalias graves en la retina.(remover)
  "nma.a",  #  2 Numero de MAs (microneurismas) encontrados en un nivel de confianza con alpha = 0.5
  "nma.b",  #  3 Numero de MAs (microneurismas) encontrados en un nivel de confianza con alpha = 0.6
  "nma.c",  #  4 Numero de MAs (microneurismas) encontrados en un nivel de confianza con alpha = 0.7
  "nma.d",  #  5 Numero de MAs (microneurismas) encontrados en un nivel de confianza con alpha = 0.8
  "nma.e",  #  6 Numero de MAs (microneurismas) encontrados en un nivel de confianza con alpha = 0.9
  "nma.f",  #  7 Numero de MAs (microneurismas) encontrados en un nivel de confianza con alpha = 1.0
  "nex.a",  #  8 Numero de Exudates encontrados en un nivel de confianza con alpha = 0.5
  "nex.b",  #  9 Numero de Exudates encontrados en un nivel de confianza con alpha = 0.6
  "nex.c",  # 10 Numero de Exudates encontrados en un nivel de confianza con alpha = 0.7
  "nex.d",  # 11 Numero de Exudates encontrados en un nivel de confianza con alpha = 0.8
  "nex.e",  # 12 Numero de Exudates encontrados en un nivel de confianza con alpha = 0.9
  "nex.f",  # 13 Numero de Exudates encontrados en un nivel de confianza con alpha = 1.0
  "nex.g",  # 14 Numero de Exudates encontrados en un nivel de confianza con alpha = 1.0
  "nex.h",  # 15 Numero de Exudates encontrados en un nivel de confianza con alpha = 1.0
  "dd",     # 16 Distancia eucladiana entre el centro de la macula y el centro del disco optico
  "dm",     # 17 Diametro del disco optico
  "amfm",   # 18 Numeri binario, relacionado con la clasificacion AM/FM-based.(remover) (indirectamente es la clase)
  "class"   # 19 Clase del dato, en donde 1 = contiene signos de DR, 0 = no contiene signos de DR.(remover)
)

########### Se convierten en factor el valor binario de class.
df.final <- df
df.final$class <- factor(df.final$class)
df.final$q <- factor(df.final$q)
df.final$ps <- factor(df.final$ps)
df.final$amfm <- factor(df.final$amfm)

#Se filtran los datos, para mantener los que son de buena calidad (q == 1)
df.final<-df.final[!(df.final$q==0),]

datos.05 <- df.final[,-c(4,5,6,7,8,10,11,12,13,14,15,16)]
datos.05.todo <- df.final[,-c(4,5,6,7,8,10,11,12,13,14,15)]
datos.06 <- df.final[,-c(3,5,6,7,8,9,11,12,13,14,15,16)]
datos.07 <- df.final[,-c(3,4,6,7,8,9,10,12,13,14,15,16)]
datos.08 <- df.final[,-c(3,4,5,7,8,9,10,11,13,14,15,16)]
datos.09 <- df.final[,-c(3,4,5,6,8,9,10,11,12,14,15,16)]
datos.10 <- df.final[,-c(3,4,5,6,7,9,10,11,12,13,15,16)]
df.final <- df.final

class <- df.final['class']






#Seleccion de caracteristicas

ranking<-InfoGainAttributeEval(class ~ . , data = datos.05) 
print(ranking)
#    nma.a       nex.a  
#0.078224967 0.035143457 
ranking<-InfoGainAttributeEval(class ~ . , data = datos.06) 
print(ranking)

# nma.b       nex.b
# 0.059602491 0.000000000
ranking<-InfoGainAttributeEval(class ~ . , data = datos.07) 
print(ranking)
#  nma.c       nex.c 
# 0.049329310 0.000000000
ranking<-InfoGainAttributeEval(class ~ . , data = datos.08) 
print(ranking)
# nma.d       nex.d 
# 0.030843628 0.000000000
ranking<-InfoGainAttributeEval(class ~ . , data = datos.09) 
print(ranking)
# nma.e       nex.e
# 0.023735965 0.028118475
ranking<-InfoGainAttributeEval(class ~ . , data = datos.10) 
print(ranking)
# nma.f       nex.f 
# 0.014917168 0.055786904

ranking<-InfoGainAttributeEval(class ~ . , data = df.final) 
print(ranking)

#Analizar ranking de caracteristicas

#########################################################
#########################################################

ranking<-InfoGainAttributeEval(class ~ . , data = datos.05) 
print(ranking)
datos.aux1 <- subset(datos.05.todo, select = -q)
datos.aux2 <- subset(datos.aux1, select = -dd)
datos.final <- subset(datos.aux2, select = -dm)

#         ps       nma.a       nex.a        amfm
# 0.004251408 0.078224967 0.035143457 0.001100858
ranking<-InfoGainAttributeEval(class ~ . , data = datos.final)
print(ranking)






## 70% of the sample size
smp_size <- floor(0.70 * nrow(datos.final))
## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(datos.final)), size = smp_size)
train <- datos.final[train_ind, ]
test <- datos.final[-train_ind, ]



#CREAR MODELO SVM (1)
#Se define la formula con las clases
formula=class ~ .
#Se pasan los parametros, y se instancia el modelo
model <- svm(formula, data = train)
print(model)
summary(model)



#PREDECIR UTIULIZANDO LOS DATOS DE PRUEBA + MODELO 
# Se pasa el modelo junto al conjunto de caracteristicas
pred <- predict(model,test)
table(pred, test$class)

#VER VALORES DE DECISION 
# compute decision values and probablities:
# Agregando probability = TRUE se puede obtener las probabilidades de pertenencia a cada una de las clases
# (No funciono) (Porfe se challaseo?)
pred <- predict(model, x, decision.values = TRUE)
attr(pred, "decision.values")[1:4,]
#attr(pred, "probability")[1:4,]

#GRAFICAR MODELO
# visualize (classes by color, SV by crosses):
plot(cmdscale(dist(train[,-6])), col = as.integer(iris[,5]), pch = c("o","+")[1:150 %in% model$index + 1])

#UTILIZACION DE TUNE CON KERNEL LINEAL
# Esta nos permite ver cuales son los hiperparametros mas adecuados para obtener el mejor modelo dentro del proceso
# de entrenamiento
# Este nos permite utilizar un esquema de validacion cruzada con el parametro CROSS
### Para un kernel lineal se utiliza el ranges list(cost = 2^(-1:4))
# Se tiene que buscar cuales son los valores mas adecuados en base al rango anterior
obj <- tune(svm, Species~., data = iris, kernel = "linear",            ranges = list(cost = 2^(-1:4)),            tunecontrol = tune.control(sampling = "cross", cross = 2 ))
summary(obj)
#El summary entrega el mejor parametro
#Mejor rendimiento
# Dispersion -> desviasion estandar asociados a los modelos de entrenamiento.

#VER COMO VARIAN LOS HIPERPARAMETROS EN FUNCION DEL ERROR
# Como es una funcion lineal, el unico hiperparametro que se esta moviendo es la funcion de costo
plot(obj)
summary(obj$best.model)

#DESEMPEÃO DEL MEJOR MODELO
pred <- predict(obj$best.model, x)
table(pred, Species)

#UTILIZACION DE TUNE CON KERNEL RADIAL
# Aqui se tiene que incorporar los hiperparametros de gamma y costo
obj <- tune(svm, Species~., data = iris, kernel = "radial", ranges = list(gamma = 2^(-2:4), cost = 2^(-1:4), tunecontrol = tune.control(sampling = "cross", cross = 2 )))
summary(obj)
pred <- predict(obj$best.model, x)
table(pred, Species)

#UTILIZACION DE TUNE CON KERNEL RADIAL (EXTENDIENDO LA BUSQUEDA DE LOS HIPERPARAMETROS)
obj <- tune(svm, Species~., data = iris, kernel = "radial", ranges = list(gamma = 2^(-7:12), cost = 2^(-7:14) , tunecontrol = tune.control(sampling = "cross", cross = 2)))
pred <- predict(obj$best.model, x)
table(pred, Species)

#Gamma nos indica el grado de linealidad, o de comportamiento lineal
# Mas bajo: Mas lineal
# Mas alto: Menos lineal


#EFICIENCIA
# Matriz de confusion














