# importing libraries
import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB4
from tinymlgen import port

# #loading saved model
# model = tf.keras.models.load_model('cifar_efficientnetb0_model.h5')

# # converting to saved_model format
# tf.saved_model.save(model,"trained_cfar_saved_model")

# #converting to tflite
# converter = tf.lite.TFLiteConverter.from_saved_model("trained_cfar_saved_model")
# tflite_model = converter.convert()

# #saving the tflite model
# with open('trained_model_cifar.tflite', 'wb') as f:
#     f.write(tflite_model)

# converting the tflite model to a C array
tf_model = tf.keras.models.load_model('cifar_efficientnetb0_model.h5')
c_code = port(tf_model)
print(c_code)
#this output that is printed needs to be put into a .h file

# #saving the tflite model
# with open('trained_model_cifar.h', 'wb') as f:
#     f.write(c_code)