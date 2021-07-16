require(e1071)
data(iris)
attach(iris)

#Apuntes:
#Es necesario dividir en datos de entrenamiento y prueba
#Es necesario realizar una seleccion de caracteristicas (eliminar las que aporten menos al modelo.)


#CREAR MODELO SVM (1)
#Se define la formula con las clases
formula=Species ~ .
#Se pasan los parametros, y se instancia el modelo
model <- svm(formula, data = iris)
print(model)
summary(model)

#Buscar info del:
# Summary del modelo
# Enfasis en: Vector de soporte

#CREAR MODELO SVM (2)
#Otra forma para instanciar el modelo (inutil)
x <- subset(iris, select = -Species)
y <- Species
model <- svm(x, y)

#PREDECIR UTIULIZANDO LOS DATOS DE PRUEBA + MODELO 
# Se pasa el modelo junto al conjunto de caracteristicas
pred <- predict(model, x)
table(pred, y)

#VER VALORES DE DECISION 
# compute decision values and probabilities:
# Agregando probability = TRUE se puede obtener las probabilidades de pertenencia a cada una de las clases
# (No funciono) (Porfe se challaseo?)
pred <- predict(model, x, decision.values = TRUE)
attr(pred, "decision.values")[1:4,]
#attr(pred, "probability")[1:4,]

#GRAFICAR MODELO
# visualize (classes by color, SV by crosses):
plot(cmdscale(dist(iris[,-5])), col = as.integer(iris[,5]), pch = c("o","+")[1:150 %in% model$index + 1])

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

#DESEMPEÑO DEL MEJOR MODELO
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



