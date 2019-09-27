from flask import Flask, render_template, request, url_for
from functions import read_images, resize_images
import cv2, numpy as np, tensorflow as tf

app = Flask('gender_predictor')

@app.route('/')
def show_predict_stock_form():
    return render_template('predictorform.html')
@app.route('/results', methods=['POST'])
def results():
    #form = request.form
    if request.method == 'POST':
      #write your function that loads the model
       model = tf.keras.models.load_model('gender_prediction_model.h5') #you can use pickle to load the trained model
       image = request.files['image']
       img_path = 'static/' + image.filename
       image.save(img_path)
       img = read_images(img_path)
       img = resize_images(img)
       prediction = model.predict(np.array([img]))[0][0]
       if prediction < 0.5:
             predicted_gender = "Female"
       elif prediction >= 0.5:
            predicted_gender = "Male"
       img_url = url_for('static', filename=image.filename)
       print(img_url)
       return render_template('resultsform.html', confidence = prediction, gender = predicted_gender, image_url = img_url)

app.run("localhost", "9999", debug=True)