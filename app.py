import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, request, jsonify
import tensorflow.lite as tflite
from openai import OpenAI
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from nltk.tokenize import word_tokenize
import pickle
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

app = Flask(__name__)

OPEN_IA_KEY = os.getenv('OPEN_AI_KEY')

# instancia del cliente de OpenIA
client = OpenAI(api_key = OPEN_IA_KEY)

def load_glove_vectors_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
# Cargar los vectores de GloVe desde el archivo binario
glove = load_glove_vectors_from_pickle('agentesProyecto\glove_vectors.pkl')

# variables para el modelo de emociones
input_shape = (96, 96, 3)
#modelo_path_img = "./emotion_recognition_model_V8.keras"
optimezed_model_img_path = "agentesProyecto\emotion_recognition_model_V8.tflite"
optimized_model_text_path = "agentesProyecto\modelo_sentiment_emotion_txt_9.tflite"


# cntexto para recibir recomendaciones  
context = "You are an expert at making recommendations based on emotions, which arise from a post, considering both the image and the accompanying text. I will provide you with two emotions: the one represented by the image and the one represented by the text. Based on these emotions, you will make recommendations such as content suggestions, words of encouragement, or actions (for example, recommending a song, a video, or a motivational quote). Also, keep in mind that I don’t want responses that are too long, and I would like you to respond with plain text, no titles, subtitles, or bold text. If you want to emphasize a word or sentence, use quotation marks. Respond in a friendly and direct manner, give the recommendation without any introduction. Also, vary your recommendations. It’s not necessary to always recommend a song or video; you can recommend them or not, and if you do, try to change it up a bit even if you receive the same two emotions multiple times."

# endpoint que recibe la imagen y texto y devuelve recomendaciones
@app.route('/recommendations', methods=['POST'])
def recommendations():
    # se obtiene la imagen y el texto
    image = request.files['image']
    text = request.form['text']

    # se obtienen las emociones de la imagen y el texto
    image_emotion = get_image_emotion(image, modelo_path=optimezed_model_img_path, input_shape=input_shape)
    text_emotion = get_text_emotion(text, modelo_path=optimized_model_text_path, glove = glove)

    if image_emotion is None:
        return jsonify("No se detecto una cara en la imagen, vuelvelo a intentar."), 400
    if text_emotion is None:
        return jsonify("No se detecto una emoción en el texto, vuelvelo a intentar."), 400

    # obtener la emoción de la imagen
    class_names_img = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'contempt']
    imgaEmo = class_names_img[image_emotion]

    # obtener la emoción del texto
    textEmo = emotion_index_to_label(text_emotion)

    # crear el prompt en base a las emociones
    prompt = f"Based on the photo, I feel {image_emotion} and based on the text of the post, I feel {textEmo}. Could you give some recommendations? try to give me new ones"

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": prompt}
        ]
    )

    # se obtienen las recomendaciones
    recommendations = completion.choices[0].message.content

    print(recommendations)

    # se devuelven las recomendaciones
    return jsonify(recommendations)
    #return jsonify(imgaEmo + " y " + textemo)

def get_image_emotion(imagen_path, modelo_path, input_shape):
    """
    Función para cargar una imagen, preprocesarla y hacer una predicción utilizando el modelo Keras.
    
    Args:
    - imagen_path: Ruta de la imagen a evaluar.
    - modelo_path: Ruta del archivo del modelo Keras (.keras).
    - input_shape: Forma de entrada del modelo (por ejemplo, (224, 224, 3)).
    
    Returns:
    - La predicción del modelo (la clase con la mayor probabilidad).
    """

    # Cargar el modelo TensorFlow Lite
    interpreter = tflite.Interpreter(model_path=modelo_path)
    interpreter.allocate_tensors()

    # Obtener detalles de los tensores de entrada y salida
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Cargar y preprocesar la imagen
    imagen = Image.open(imagen_path)  # Cargar la imagen

    # preparar la imagen para el modelo
    face = detect_face(imagen)

    #prepared_image = prepare_image_for_model(face)
    if(face is None):
        #raise ValueError("No se detecto una cara en la imagen, vuelvelo a intentar.")
        return None
    
    # preparar la imagen para realizar la prediccion
    prepared_image = prepare_image_for_model(face)
    # Convertir la imagen preparada a float32
    prepared_image = prepared_image.astype('float32')

    # Asignar la imagen al tensor de entrada del modelo
    interpreter.set_tensor(input_details[0]['index'], prepared_image)

    # Realizar la predicción
    interpreter.invoke()

    # Obtener el resultado de salida
    prediccion = interpreter.get_tensor(output_details[0]['index'])

    # Obtener la clase con la mayor probabilidad
    clase_predicha = np.argmax(prediccion, axis=1)[0]
    
    return clase_predicha

def get_text_emotion(text, modelo_path, glove):
    # Cargar el modelo TensorFlow Lite
    interpreter = tflite.Interpreter(model_path=modelo_path)
    interpreter.allocate_tensors()

    # Obtener detalles de los tensores de entrada y salida
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocesar la oración
    preprocessed_sentence = preprocess_text(text)
    tokenized_sentence = word_tokenize(preprocessed_sentence)
    sentence_vector = sentence_to_vector(tokenized_sentence, glove)

    # Obtener la forma y el tipo de datos esperados del tensor de entrada
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    # Asegurarse de que sentence_vector tenga la forma y el tipo de datos correctos
    if sentence_vector.shape != tuple(input_shape[1:]):
        sentence_vector = np.reshape(sentence_vector, tuple(input_shape[1:]))

    if sentence_vector.dtype != input_dtype:
        sentence_vector = sentence_vector.astype(input_dtype)

    # Replicar el tensor para que tenga el tamaño de lote esperado
    sentence_vector = np.tile(sentence_vector, (input_shape[0], 1))

    # Establecer el tensor de entrada
    interpreter.set_tensor(input_details[0]['index'], sentence_vector)

    # Ejecutar la inferencia
    interpreter.invoke()

    # Obtener la salida del modelo
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(output_data, axis=1)[0]  # Obtener el índice de la emoción predicha

    return predicted_label  # Retorna solo el índice de la emoción predicha

# funcion que detecta la cara en una imagen
def detect_face(image):
    #image = cv2.imread(image_path)
    # Convertir la imagen de PIL a un array de NumPy
    image = np.array(image)
    # Convertir la imagen de RGB a BGR (formato que usa OpenCV)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    roi_gray = None
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = image[y:y+h, x:x+w]

    if roi_gray is None:
        #raise ValueError("No face detected in the image")
        return None

    return roi_gray

# funcion de preparacion de imagen para el modelo
def prepare_image_for_model(face_image):
    resized = cv2.resize(face_image, (96, 96), interpolation=cv2.INTER_AREA)
    img_result = np.expand_dims(resized, axis=0)  # Add batch dimension
    img_result = img_result / 255.0  # Normalize the image
    return img_result

def preprocess_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = ''.join([char if char.isalnum() or char.isspace() else ' ' for char in text])  # Eliminar puntuación
    return text

def sentence_to_vector(sentence, glove):
    vectors = [glove[word] for word in sentence if word in glove]
    if len(vectors) == 0:
        return np.zeros(300)  # Vector vacío si no hay palabras en GloVe
    return np.mean(vectors, axis=0)
    
def save_glove_vectors(glove, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(glove, f)

# Función para convertir el índice a la etiqueta de la emoción
def emotion_index_to_label(index):
    emotion_labels = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
    return emotion_labels[index]

if __name__ == '__main__':
    app.run(debug=True)