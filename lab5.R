
#install.packages('bnlearn')
require(e1071)
require(RWeka)
require(ggplot2)
####################################################################################
####################################################################################
####################################################################################
######### NUESTRO #########
####################################################################################
####################################################################################
####################################################################################

#OJO: Es importante estandarizar las variables.
#"sí hay que estandarizarlos, de lo contrario, los predictores de mayor magnitud eclipsarán a los de menor magnitud."
#https://www.cienciadedatos.net/documentos/34_maquinas_de_vector_soporte_support_vector_machines


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
datos.aux1 <- subset(datos.05, select = -q)
datos.aux2 <- subset(datos.aux1, select = -dd)
datos.aux3 <- subset(datos.aux2, select = -dm)

#         ps       nma.a       nex.a        amfm
# 0.004251408 0.078224967 0.035143457 0.001100858
ranking<-InfoGainAttributeEval(class ~ . , data = datos.aux3)
print(ranking)


datos.aux3$nma.a <- scale(datos.aux3$nma.a)
datos.aux3$nex.a <- scale(datos.aux3$nex.a)
ps <- data.frame(datos.aux3$ps)
amfm <- data.frame(datos.aux3$amfm)
class <- data.frame(datos.aux3$class)
aux.1 <- subset(datos.aux3, select = -class)
aux.2 <- subset(aux.1, select = -ps)
aux.3 <- subset(aux.2, select = -amfm)
aux.4 <- scale(aux.3)

datos.final <- datos.aux3

## 70% of the sample size
smp_size <- floor(0.70 * nrow(datos.final))
## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(datos.final)), size = smp_size)
train <- datos.final[train_ind, ]
test <- datos.final[-train_ind, ]


##########################################################################################
############################## MODELO SVM SIN CAMBIOS ####################################
##########################################################################################
#Se define la formula con las clases
formula=class ~ .
#Se pasan los parametros, y se instancia el modelo
set.seed(100)
model <- svm(formula = class ~ ., data = train)
print(model)
summary(model)
plot(cmdscale(dist(train[,-5])), col = as.integer(train[,5]), pch = c("o","+")[1:150 %in% model$index + 1])



############################################
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
############################################

#UTILIZACION DE TUNE CON KERNEL LINEAL
# Esta nos permite ver cuales son los hiperparametros mas adecuados para obtener el mejor modelo dentro del proceso
# de entrenamiento
# Este nos permite utilizar un esquema de validacion cruzada con el parametro CROSS
### Para un kernel lineal se utiliza el ranges list(cost = 2^(-1:4))
# Se tiene que buscar cuales son los valores mas adecuados en base al rango anterior


# ranges: Valores de hiperparametros que se van a evaluar
# cost: Hiperparametro de penalizacion, este norma el balance entre el bias y varianza del modelo
# Este afecta en la eleccion de los vectores de soporte.
# C = 0 -> divicion perfecta (0 erorres) [Puede traer errores de overfitting]

# Es necesario utulizar las maquinas de vector de soporte cuando no se puede dividir linealmente.
# Esto consiste en expandir las dimenciones del espacio original.



##########################################################################################
############################## MODELO SVM CON KERNEL LINEAL ##############################
##########################################################################################
set.seed(100)
obj <- tune(svm, class~., data = datos.final, kernel = "linear",ranges = list(cost = 2^(-4:4)),tunecontrol = tune.control(sampling = "cross", cross = 2 ))
summary(obj)

obj$best.model

summary(obj$best.model)
#El summary entrega el mejor parametro
#Mejor rendimiento
# Dispersion -> desviasion estandar asociados a los modelos de entrenamiento.

#VER COMO VARIAN LOS HIPERPARAMETROS EN FUNCION DEL ERROR
# Como es una funcion lineal, el unico hiperparametro que se esta moviendo es la funcion de costo
plot(obj)
summary(obj$best.model)
model <- obj$best.model

plot(cmdscale(dist(datos.final[,-5])), col = as.integer(datos.final[,5]), pch = c("o","+")[1:150 %in% model$index + 1])


#DESEMPEO DEL MEJOR MODELO
pred <- predict(obj$best.model, datos.final[,-5])
conf.matrix <- table(pred, datos.final[,5])

VP <- conf.matrix[1]
FP <- conf.matrix[3]
VN <- conf.matrix[4]
FN <- conf.matrix[2]

precision = VP / (VP + FP)
recall = VP / (VP + FN)
calculoF1 <- 2*precision*recall/(precision + recall)

conf.matrix
precision
recall
calculoF1



##########################################################################################
############################## MODELO SVM CON KERNEL RADIAL ##############################
##########################################################################################

#UTILIZACION DE TUNE CON KERNEL RADIAL
# Aqui se tiene que incorporar los hiperparametros de gamma y costo
set.seed(100)
obj <- tune(svm, class~., data = datos.final, kernel = "radial", ranges = list(gamma = 2^(-2:3), cost = 2^(-3:1), tunecontrol = tune.control(sampling = "cross", cross = 2 )))
summary(obj)

plot(obj)
summary(obj$best.model)
model <- obj$best.model

plot(cmdscale(dist(datos.final[,-5])), col = as.integer(datos.final[,5]), pch = c("o","+")[1:150 %in% model$index + 1])


set.seed(423)
obj1 <- tune(svm, class~., data = datos.final, kernel = "radial", ranges = list(gamma = 2^(-5:5), cost = 2^(2:7) , tunecontrol = tune.control(sampling = "cross", cross = 2)))
summary(obj1)

set.seed(423)
obj2 <- tune(svm, class~., data = datos.final, kernel = "radial", ranges = list(gamma = 2^(-7:2), cost = 2^(5:8) , tunecontrol = tune.control(sampling = "cross", cross = 2)))
summary(obj2)

set.seed(423)
obj3 <- tune(svm, class~., data = datos.final, kernel = "radial", ranges = list(gamma = 2^(-7:2), cost = 2^(-3:2) , tunecontrol = tune.control(sampling = "cross", cross = 2)))
summary(obj3)

set.seed(423)
obj4 <- tune(svm, class~., data = datos.final, kernel = "radial", ranges = list(gamma = 2^(-5:2), cost = 2^(1:5) , tunecontrol = tune.control(sampling = "cross", cross = 2)))
summary(obj4)




ggplot(data = obj1$performances, aes(x = cost, y = error, color = as.factor(gamma)))+
  geom_line() +
  geom_point() +
  labs(title = "Error de clasificación vs hiperparámetros C y gamma", color = "gamma") +
  theme_bw() +
  theme(legend.position = "bottom")
summary(obj1)


ggplot(data = obj2$performances, aes(x = cost, y = error, color = as.factor(gamma)))+
  geom_line() +
  geom_point() +
  labs(title = "Error de clasificación vs hiperparámetros C y gamma", color = "gamma") +
  theme_bw() +
  theme(legend.position = "bottom")
summary(obj2)

ggplot(data = obj3$performances, aes(x = cost, y = error, color = as.factor(gamma)))+
  geom_line() +
  geom_point() +
  labs(title = "Error de clasificación vs hiperparámetros C y gamma", color = "gamma") +
  theme_bw() +
  theme(legend.position = "bottom")
summary(obj3)

ggplot(data = obj4$performances, aes(x = cost, y = error, color = as.factor(gamma)))+
  geom_line() +
  geom_point() +
  labs(title = "Error de clasificación vs hiperparámetros C y gamma", color = "gamma") +
  theme_bw() +
  theme(legend.position = "bottom")
summary(obj4)


model <- obj3$best.model
summary(obj3$best.model)
model <- obj3$best.model
plot(cmdscale(dist(datos.final[,-5])), col = as.integer(datos.final[,5]), pch = c("o","+")[1:150 %in% model$index + 1])

#DESEMPEO DEL MEJOR MODELO
pred <- predict(model, datos.final[,-5])
conf.matrix <- table(pred, datos.final[,5])

VP <- conf.matrix[1]
FP <- conf.matrix[3]
VN <- conf.matrix[4]
FN <- conf.matrix[2]

precision = VP / (VP + FP)
recall = VP / (VP + FN)
calculoF1 <- 2*precision*recall/(precision + recall)

conf.matrix
precision
recall
calculoF1






#Gamma nos indica el grado de linealidad, o de comportamiento lineal
# Mas bajo: Mas lineal
# Mas alto: Menos lineal


#EFICIENCIA
# Matriz de confusion














