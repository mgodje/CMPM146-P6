from models.model import Model
from tensorflow.keras import Sequential, layers, Input
from tensorflow.keras.layers import Rescaling, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        

        self.model = Sequential([
            Input(shape=input_shape),
            Rescaling(1./255),

           #Convolutional layers
            layers.Conv2D(10, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(19, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),

           #Fully connected layer
            layers.Flatten(),
            layers.Dense(6, activation="relu"),
            # layers.Dropout(0.5),
            layers.Dense(categories_count, activation="softmax")
        
            ])
        
        
        # you have to initialize self.model to a keras model

    
    def _compile_model(self):
        # Your code goes here

        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        # you have to compile the keras model, similar to the example in the writeup
  