from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class RandomModel(Model):
    def _define_model(self, input_shape, categories_count):
        old_model = models.load_model('results/70_percent_accuracy.keras') #old model
       
        old_model = Sequential(old_model.layers[:-1])

        for layer in old_model.layers:
            layer.trainable = False

            

        self.model = Sequential([
            old_model, 
            layers.Dense(39, activation="relu"), 
            layers.Dense(categories_count, activation="softmax")
        ])
    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        self.model.compile(
            optimizer=Adam(learning_rate=0.001), 
            loss='categorical_crossentropy', 
            metrics=['accuracy'])

    @staticmethod
    def _randomize_layers(model):
        # Your code goes here

        # you can write a function here to set the weights to a random value
        # use this function in _define_model to randomize the weights of your loaded model
        for layer in model.layers:
            # randomize the weights of the loaded model
            RandomModel._randomize_layer(layer)
