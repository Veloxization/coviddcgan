from keras.models import model_from_json
from keras.utils import plot_model

model_name = "covidmodel"

f = open(f"saved_model/{model_name}/generator.json")
generator_data = f.read()
generator = model_from_json(generator_data)
f.close()

f = open(f"saved_model/{model_name}/discriminator.json")
discriminator_data = f.read()
discriminator = model_from_json(discriminator_data)
f.close()

f = open(f"saved_densenet/real_dataset/densenet.json")
densenet_data = f.read()
densenet = model_from_json(densenet_data)
f.close()

plot_model(generator, to_file="generator.png", show_shapes=False, show_layer_names=False, expand_nested=True)
plot_model(discriminator, to_file="discriminator.png", show_shapes=False, show_layer_names=False, expand_nested=True)
plot_model(densenet, to_file="densenet.png", show_shapes=False, show_layer_names=False, expand_nested=False)
