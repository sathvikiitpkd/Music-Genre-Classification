import streamlit as st
import librosa
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import load_img, img_to_array
st.title('Music Genre Classification Predication')
import librosa.display
import numpy as np
import matplotlib. pyplot as plt
import pydub
from pathlib import Path
uploaded_file = st.file_uploader("upload", type=['wav'])
audio = pydub.AudioSegment.from_wav(uploaded_file)
save_path = Path("temp") / uploaded_file.name
audio.export(save_path, format='wav')
import librosa
import librosa.display
y, sr = librosa.load(save_path)
whale_song, _ = librosa.effects.trim(y)
import numpy as np
import matplotlib.pyplot as plt
n_fft = 2048
hop_length = 512
D = np.abs(librosa.stft(whale_song, n_fft=n_fft,  hop_length=hop_length))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear');
plt.colorbar(format='%+1.0f dB')
plt.savefig("./ans/out.png")
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
loaded_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['accuracy'])
import os
def load_images_from_path(path, label):
    images = []
    labels = []

    for file in os.listdir(path):
        images.append(img_to_array(load_img(os.path.join(path, file), target_size=(224, 224, 3))))
        labels.append((label))
        
    return images, labels
i, l = load_images_from_path('./ans', 1) 
prediction = loaded_model.predict(np.array(i)/255)
ans = np.argmax(prediction[0],axis=0)
def numbers_to_strings(argument):
    switcher = {
        0: "blues",
        1: "classical",
        2: "country",
        3: "disco",
        4: "hiphop",
        5: "jazz",
        6: "metal",
        7: "pop",
        8: "reggae",
        9: "rock",
    }
    return switcher.get(argument, "nothing")

if __name__ == "__main__":
    st.write(prediction)
    st.write("Music Genre for above audio is ", numbers_to_strings(ans))
