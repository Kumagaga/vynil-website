import pathlib
from os.path import join, isfile, dirname
from os import listdir
from itertools import cycle
import streamlit as st
import numpy as np
from PIL import Image
from skimage import transform
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


st.set_page_config(page_title="CNN to classify album image")

STREAMLIT_STATIC_PATH = (
    pathlib.Path(st.__path__[0]) / "static"
)

st.title("CNN to classify album image")
st.subheader(
    "Deep learning model that classifies album covers\
    on onine marketplaces with more than 65% confidence."
)
st.write(
    "The objective was to create a tool that could quickly and accurately classify album covers.\
    One application for this is to automate a purchaser bot that scans marketplaces and buys\
    albums that are underpriced."
    )
st.write(
    "If you'd like to learn more, please check our project on \
    [GitHub](https://github.com/nihonlanguageprocessing/vynil_id)"
)

# Returns the file names for example images
file_dir = join(dirname(__file__), "test_img")
file_names = [f for f in listdir(file_dir) if isfile(join(file_dir, f))]

# Creates a list with example images
img_list = []
for img in file_names:
    img_path = join(file_dir, img)
    to_image = Image.open(img_path)
    img_list.append(to_image)

# Displays and captions example images
st.subheader("Representative images of 6 album covers")
st.write(
    "The current model has only been trained on a set of albums we've curated.\
    Note that images below are from Mercari and were not used to train the model."
)

cols = cycle(st.columns(3))
for label, img in enumerate(img_list):
    next(cols).image(img, width=150, caption=file_names[label].replace('.jpg', ''))



def load_img(jpg):
    # Preprocesses the jpg prior to making predictions
    np_image = Image.open(jpg)
    np_image = img_to_array(np_image)
    np_image = transform.resize(np_image, (150, 150, 3))
    np_image /= 255.0
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def predict(img):
    # Predicts the species of the plant in img
    model = load_model("model_vynil_22-8-27.h5", compile=False)
    result = model.predict(img)
    return result

def process_predict(result):
    # Assigns a label to the top prediction
    # Keras models return predictions in alphabetical order of original labels
    labels = [
            'Anri (2) - Bi・Ki・Ni', #0
            'Anri (2) - Coool', #1
            'Anri (2) - Timely!!', #2
            'Anri (2) - Wave', #3
            'Hiroshi Sato - Aqua', #4
            'Hiroshi Sato - Awakening', #5
            'Hiroshi Sato - Future File', #6
            'Hiroshi Sato - Sound Of Science', #7
            'Mariya Takeuchi - Beginning', #8
            'Mariya Takeuchi - Love Songs', #9
            'Mariya Takeuchi - Miss M', #10
            'Mariya Takeuchi - Request', #11
            'Mariya Takeuchi - Trad = トラッド', #12
            'Mariya Takeuchi - University Street', #13
            'Mariya Takeuchi - Variety', #14
            'Momoko Kikuchi - Adventure', #15
            'Momoko Kikuchi - Escape From Dimension', #16
            'Momoko Kikuchi - Ocean Side', #17
            'Momoko Kikuchi - Tropic Of Capricorn =トロピック・オブ・カプリコーン 南回帰線', #18
            'Taeko Ohnuki - Aventure', #19
            'Taeko Ohnuki - Cliché', #20
            'Taeko Ohnuki - Grey Skies', #21
            'Taeko Ohnuki - Mignonne', #22
            'Taeko Ohnuki - Romantique', #23
            'Taeko Ohnuki - Sunshower', #24
            'Tatsuro Yamashita - Big Wave = ビッグウェイブ', #25
            'Tatsuro Yamashita - Circus Town', #26
            'Tatsuro Yamashita - For You', #27
            'Tatsuro Yamashita - Go Ahead!', #28
            'Tatsuro Yamashita - Greatest Hits! Of', #29
            "Tatsuro Yamashita - It's A Poppin' Time", #30
            'Tatsuro Yamashita - Melodies', #31
            'Tatsuro Yamashita - Moonglow', #32
            'Tatsuro Yamashita - On The Street Corner', #33
            'Tatsuro Yamashita - On The Street Corner 2', #34
            'Tatsuro Yamashita - Ray Of Hope', #35
            'Tatsuro Yamashita - Ride On Time', #36
            'Tatsuro Yamashita - Softly', #37
            'Tatsuro Yamashita - Spacy', #38
            'Utada Hikaru - Badモード', #39
            'Utada Hikaru - Deep River', #40
            'Utada Hikaru - First Love', #41
            '東北新幹線 - Thru Traffic', #42
    ]
    # Creates a dictionary matching predictions with specie s
    predictions = dict(zip(labels, result[0]))
    return predictions

# Allows users to upload and image
st.header("Please try by uploading dragging and dropping one of the images above or uploading your own image directly (.jpeg).")
jpg = st.file_uploader("Upload an album cover image", type=['jpg','jpeg'])

# The model makes predictions and displays them
if jpg:
    img = load_img(jpg)
    images = np.vstack([img])
    result = predict(img)
    predictions = process_predict(result)
    model_result = round(result[0][np.argmax(result)]*100, 2)
    st.header("Your Results")
    st.empty().image(jpg)
    st.write('File name: ' + jpg.name)
    if result[0][np.argmax(result)]*100 <= 65:
        st.write("Prediction confidence rate below 65%")
        st.write('Undetectable album image.')

    elif np.argmax(result) == 0:
        st.write("Album name: Anri (2) - Bi・Ki・Ni")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 1:
        st.write("Album name: Anri (2) - Coool")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 2:
        st.write("Album name: Anri (2) - Timely!!")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 3:
        st.write("Album name: Anri (2) - Wave")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 4:
        st.write("Album name: Hiroshi Sato - Aqua")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 5:
        st.write("Album name: Hiroshi Sato - Awakening")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 6:
        st.write("Album name: Hiroshi Sato - Future jpg.name")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 7:
        st.write("Album name: Hiroshi Sato - Sound Of Science")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 8:
        st.write("Album name: Mariya Takeuchi - Beginning")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 9:
        st.write("Album name: Mariya Takeuchi - Love Songs")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 10:
        st.write("Album name: Mariya Takeuchi - Miss M")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 11:
        st.write("Album name: Mariya Takeuchi - Request")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 12:
        st.write("Album name: Mariya Takeuchi - Trad = トラッド")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 13:
        st.write("Album name: Mariya Takeuchi - University Street")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 14:
        st.write("Album name: Mariya Takeuchi - Variety")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 15:
        st.write("Album name: Momoko Kikuchi - Adventure")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 16:
        st.write("Album name: Momoko Kikuchi - Escape From Dimension")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 17:
        st.write("Album name: Momoko Kikuchi - Ocean Side")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 18:
        st.write("Album name: Momoko Kikuchi - Tropic Of Capricorn =トロピック・オブ・カプリコーン 南回帰線")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 19:
        st.write("Album name: Taeko Ohnuki - Aventure")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 20:
        st.write("Album name: Taeko Ohnuki - Cliché")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 21:
        st.write("Album name: Taeko Ohnuki - Grey Skies")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 22:
        st.write("Album name: Taeko Ohnuki - Mignonne")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 23:
        st.write("Album name: Taeko Ohnuki - Romantique")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 24:
        st.write("Album name: Taeko Ohnuki - Sunshower")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 25:
        st.write("Album name: Tatsuro Yamashita - Big Wave = ビッグウェイブ")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 26:
        st.write("Album name: Tatsuro Yamashita - Circus Town")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 27:
        st.write("Album name: Tatsuro Yamashita - For You")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 28:
        st.write("Album name: Tatsuro Yamashita - Go Ahead!")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 29:
        st.write("Album name: Tatsuro Yamashita - Greatest Hits! Of")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 30:
        st.write("Album name: Tatsuro Yamashita - It's A Poppin' Time")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 31:
        st.write("Album name: Tatsuro Yamashita - Melodies")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 32:
        st.write("Album name: Tatsuro Yamashita - Moonglow")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 33:
        st.write("Album name: Tatsuro Yamashita - On The Street Corner")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 34:
        st.write("Album name: Tatsuro Yamashita - On The Street Corner 2")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 35:
        st.write("Album name: Tatsuro Yamashita - Ray Of Hope")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 36:
        st.write("Album name: Tatsuro Yamashita - Ride On Time")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 37:
        st.write("Album name: Tatsuro Yamashita - Softly")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 38:
        st.write("Album name: Tatsuro Yamashita - Spacy")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 39:
        st.write("Album name: Utada Hikaru - Badモード")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 40:
        st.write("Album name: Utada Hikaru - Deep River")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 41:
        st.write("Album name: Utada Hikaru - First Love")
        st.write("Confidence: " + str(model_result) + "%")

    elif np.argmax(result) == 42:
        st.write("Album name: 東北新幹線 - Thru Traffic")
        st.write("Confidence: " + str(model_result) + "%")

st.write('-----------------------------------------------')
st.subheader('The list of predictable album covers')
st.write('1. Anri (2) - Bi・Ki・Ni')
st.write('2. Anri (2) - Coool')
st.write('3. Anri (2) - Timely!!')
st.write('4. Anri (2) - Wave')
st.write('5. Hiroshi Sato - Aqua')
st.write('6. Hiroshi Sato - Awakening')
st.write('7. Hiroshi Sato - Future File')
st.write('8. Hiroshi Sato - Sound Of Science')
st.write('9. Mariya Takeuchi - Beginning')
st.write('10. Mariya Takeuchi - Love Songs')
st.write('11. Mariya Takeuchi - Miss M')
st.write('12. Mariya Takeuchi - Request')
st.write('13. Mariya Takeuchi - Trad = トラッド')
st.write('14. Mariya Takeuchi - University Street')
st.write('15. Mariya Takeuchi - Variety')
st.write('16. Momoko Kikuchi - Adventure')
st.write('17. Momoko Kikuchi - Escape From Dimension')
st.write('18. Momoko Kikuchi - Ocean Side')
st.write('19. Momoko Kikuchi - Tropic Of Capricorn =トロピック・オブ・カプリコーン 南回帰線')
st.write('20. Taeko Ohnuki - Aventure')
st.write('21. Taeko Ohnuki - Cliché')
st.write('22. Taeko Ohnuki - Grey Skies')
st.write('23. Taeko Ohnuki - Mignonne')
st.write('24. Taeko Ohnuki - Romantique')
st.write('25. Taeko Ohnuki - Sunshower')
st.write('26. Tatsuro Yamashita - Big Wave = ビッグウェイブ')
st.write('27. Tatsuro Yamashita - Circus Town')
st.write('28. Tatsuro Yamashita - For You')
st.write('29. Tatsuro Yamashita - Go Ahead!')
st.write('30. Tatsuro Yamashita - Greatest Hits! Of')
st.write("31. Tatsuro Yamashita - It's A Poppin' Time")
st.write('32. Tatsuro Yamashita - Melodies')
st.write('33. Tatsuro Yamashita - Moonglow')
st.write('34. Tatsuro Yamashita - On The Street Corner')
st.write('35. Tatsuro Yamashita - On The Street Corner 2')
st.write('36. Tatsuro Yamashita - Ray Of Hope')
st.write('37. Tatsuro Yamashita - Ride On Time')
st.write('38. Tatsuro Yamashita - Softly')
st.write('39. Tatsuro Yamashita - Spacy')
st.write('40. Utada Hikaru - Badモード')
st.write('41. Utada Hikaru - Deep River')
st.write('42. Utada Hikaru - First Love')
st.write('43. 東北新幹線 - Thru Traffic')
