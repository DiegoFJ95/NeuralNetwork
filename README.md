# NeuralNetwork

Correr NeuralNetwork.py para el modelo sin framework y framework.py para el modelo con framework

Puedes encontrar el [reporte detallado aquí.](https://drive.google.com/file/d/1C_z7pexeze1pXaJ9Iq-cBmDfxiYWDCf7/view?usp=sharing)

Por defecto el modelo predice el nivel de estrés según los datos de la autoestima, la calidad de sueño, la depresión, el nivel de ansiedad, y la frecuencia de dolor de cabeza.

Se usé este [dataset](https://www.kaggle.com/datasets/mdsultanulislamovi/student-stress-monitoring-datasets/data)


### Características del modelo:

El modelo se entrena durante 200 generaciones con un learning rate de 0.01 y suele llegar a un error de 0.2 para el entrenamiento y 0.4 para las pruebas si no se atora en un minimo local.

Al principio entrena un modelo pequeño para verificar que la red neuronal esté bien programada, debe dar como resultados 1, -1, -1, y 1.

Usa TanH como función de activación final y ReLU en las capas ocultas.

Usa el mean squared error como función de pérdida. 

### Consideración especial:


Aumenté el límite de llamadas recursivas para poder hacer backpropagation en modelos grandes, ya que la implementación con iteradores en lugar de recursividad hacía el modelo muchísimo más lento. Esto aumenta el riesgo de que haya stack overflow, aún estoy bsucando otra solución.
