from google.cloud import storage
import tensorflow as tf
import numpy as np

# Define the bucket name and class names
BUCKET_NAME = 'dog-breed-model-1' 
CLASS_NAMES = ['Chihuahua', 'Japanese Spaniel', 'Maltese Dog', 'Pekinese', 'Shih-Tzu', 'Blenheim_spaniel', 'Papillon', 'Toy Terrier', 'Rhodesian Ridgeback', 'Afghan Hound', 'Basset', 'Beagle', 'Bloodhound', 'Bluetick', 'Black and Tan Coonhound', 'Walker Hound', 'English Foxhound', 'Redbone', 'Borzoi', 'Irish Wolfhound', 'Italian Greyhound', 'Whippet', 'Ibizan Hound', 'Norwegian Elkhound', 'Otterhound', 'Saluki', 'Scottish Deerhound', 'Weimaraner', 'Staffordshire Bullterrier', 'American Staffordshire Terrier', 'Bedlington Terrier', 'Border Terrier', 'Kerry blue Terrier', 'Irish Terrier', 'Norfolk  Terrier', 'Norwich Terrier', 'Yorkshire Terrier', 'Wire haired Fox Terrier', 'Lakeland Terrier', 'Sealyham Terrier', 'Airedale Terrier', 'Cairn Terrier', 'Australian Terrier', 'Dandie Dinmont', 'Boston Bulldog', 'miniature Schnauzer', 'giant Schnauzer', 'standard Schnauzer', 'Scotch terrier', 'Tibetan terrier', 'Silky terrier', 'soft coated Wheaten terrier', 'West Highland white terrier', 'Lhasa', 'flat coated retriever', 'curly coated retriever', 'Golden Retriever', 'Labrador retriever', 'Chesapeake Bay retriever', 'German short haired pointer', 'Vizsla', 'English setter', 'Irish setter', 'Gordon setter', 'Brittany spaniel', 'Clumber Spaniel', 'English springer', 'Welsh springer spaniel', 'cocker Spaniel', 'Sussex Spaniel', 'Irish Water Spaniel', 'Kuvasz', 'Schipperke', 'Groenendael', 'Malinois', 'Briard', 'Australian Kelpie', 'Komondor', 'Old English Sheepdog', 'Shetland Sheepdog', 'Collie', 'Border Collie', 'Bouvier Des Flandres', 'Rottweiler', 'German_shepherd', 'Doberman', 'Miniature Pinscher', 'Greater Swiss Mountain dog', 'Bernese mountain dog', 'Appenzeller', 'EntleBucher', 'Boxer', 'Bull Mastiff', 'Tibetan mastiff', 'French Bulldog', 'Great Dane', 'Saint Bernard', 'Eskimo_dog', 'Malamute', 'Siberian Husky', 'Affenpinscher', 'Basenji', 'Pug', 'Leonberg', 'Newfoundland', 'Great_Pyrenees', 'Samoyed', 'Pomeranian', 'Chow', 'Keeshond', 'Brabancon Griffon', 'Pembroke', 'Cardigan', 'Toy Poodle', 'Miniature Poodle', 'Standard Poodle', 'Mexican Hairless Dog', 'Dingo', 'Dhole', 'African hunting dog']
model = None

def download_blob(bucket_name, source_blob_name, destination_file_name):
    '''
    Downloads a blob from the bucket
    '''
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def reshape_str(array_str):
    '''	
    Reshapes the string to a numpy array
    '''
    array_str = array_str.split(',')
    array_floats = list(map(float, array_str))
    array_np = np.array(array_floats)
    image = array_np.reshape((256, 256, 3))
    image = np.expand_dims(image, axis=0)
    return image


def predict(request):
    '''
    Predicts the class of the image
    '''
    global model
    if model is None:
        # Download the model from the bucket if it is not loaded
        download_blob(BUCKET_NAME,
                    'models/SGD-0.01-0.99-20epochs.h5',
                        '/tmp/SGD-0.01-0.99-20epochs.h5')
        # Load the model
        model = tf.keras.models.load_model("/tmp/SGD-0.01-0.99-20epochs.h5")

    # Get the image from the request
    data = request.form.to_dict()
    array_str = data.get("array")
    # Reshape the string to a numpy array
    image = reshape_str(array_str)
    # Predict the class
    predictions = model.predict(image)
    # Get the class name and confidence
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return {"class": predicted_class, "confidence": confidence}
