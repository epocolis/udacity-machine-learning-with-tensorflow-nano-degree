import argparse
from unicodedata import category
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tensorflow as tf
from keras_preprocessing import image 
import numpy as np
import pprint


IMAGE_SIZE = (224, 224)
TF_DATASET_NAME = 'oxford_flowers102'

parser = argparse.ArgumentParser(description="Flowers Classifier")
parser.add_argument("image_path", action="store", type=str)
parser.add_argument("model_path", action="store")

parser.add_argument(
    "--top_k",
    action="store",
    type=int,
    default=argparse.SUPPRESS,
    help="Returns the Top K predictions",
)


parser.add_argument(
    "--category_names",
    action="store",
    default=argparse.SUPPRESS,
    help="The path to a label_map.json file that contains label -> class_name mappings",
)

def load_model(model_path):
    model = tf.keras.models.load_model((model_path),custom_objects={'KerasLayer':hub.KerasLayer})
    return model

def process_image(image):
    image = np.asarray(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE)
    image /= 255
    image = np.expand_dims(image, axis=0)
    return image    

def load_image(image_path):
    im = image.load_img(image_path, target_size=IMAGE_SIZE)
    input_image = process_image(im)
    return input_image

def predict(im, model, top): 
    _,dataset_info = tfds.load(TF_DATASET_NAME, with_info=True,download=False)
    preds = model.predict(im).squeeze()
    top_indices = preds.argsort()[-top:][::-1]
    result = {dataset_info.features['label'].names[i] : preds[i] for i in top_indices}
    return result


def  predict(model,image, category_names_dict=None, top_k=1):
    preds = model.predict(image).squeeze()
    if top_k == "1":
        top_pred_idx = preds.argsort()[0]
        result = {dataset_info.features['label'].names[top_pred_idx]: preds[top_pred_idx]}
        
    else:

        top_indices = preds.argsort()[-top_k:][::-1]
        if category_names_dict: 
            result = {category_names_dict[i] : preds[i] for i in top_indices}
        else: 
            # look it up ourselves 
            _,dataset_info = tfds.load('oxford_flowers102', with_info=True,download=False)   
            result = {dataset_info.features['label'].names[i] : preds[i] for i in top_indices}

    return result

def load_category_names(category_names_path):
    category_names_dict = {}
    return category_names_dict


def main():
    args = parser.parse_args()
    # get the required arguments
    model_path = args.model_path
    image_path = args.image_path

    model = load_model(model_path)
    img =   load_image(image_path)
    


    if "top_k" in args and 'category_names' in args: 
        # then 
        top_k = args.top_k
        category_names_path = args.category_names
        category_names_dict = load_category_names(category_names_path)
        preds = predict(model,img, category_names_dict,top_k)
       
    if "top_k" not in args and 'category_names' in args: 
        category_names_path = args.category_names
        category_names_dict = load_category_names(category_names_path)
        preds = predict(model,img, category_names_dict)

    if "top_k" not in args and 'category_names' not in args:  
         preds = predict(model,img)    
        

    pp = pprint.PrettyPrinter(width=41, compact=True)
    pp.pprint(preds)

if __name__ == "__main__":
    main()
