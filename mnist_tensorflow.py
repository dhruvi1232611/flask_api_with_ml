import tensorflow as tf
import keras

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()

x_train=x_train/255.0
x_test=x_test/255.0

model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dense(64,activation="softmax")
])

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))
model.predict(x_test)
test_loss,test_accuracy=model.evaluate(x_test,y_test)
print("Accuracy:",test_accuracy)
print("test_loss:",test_loss)
model.save("save_model/tensor.keras")
print("Tensorflow model saved")