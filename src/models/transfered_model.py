from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class TransferedModel(Model):
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

    # from basic_model
    def _compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=0.001), 
            loss='categorical_crossentropy', 
            metrics=['accuracy'])
