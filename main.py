import cv2 as cv
import numpy as np
import streamlit as st
import pandas as pd
import requests
from darknet import DarkNetwork

table_network = DarkNetwork('models/darknet/yolo_tiny_monitor.cfg',
                            'models/darknet/yolo_tiny_monitor_last.weights',
                            'models/darknet/classes.names',
                            probability_minimum=0.7)
def telegram_bot_sendtext(bot_message):

   bot_token = '1419426197:AAHA44__TBtasdgmxNSFZoqHH4OzE18Md7U'
   bot_chatID = '310764709'
   send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

   response = requests.get(send_text)

   return response.json()

def decodeImage(data):


    #Gives us 1d array
    decoded = np.fromstring(data, dtype=np.uint8)
    #We have to convert it into (270, 480,3) in order to see as an image
    decoded = decoded.reshape((270, 480,3))
    return decoded;

def page_picture():

    html_upload = """
         <h2 align="center">Upload picture to predict</h2>
    """
    st.markdown(html_upload, unsafe_allow_html=True)
    image = cv.imread('images/1.png'.format())
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    table_network.get_network_result(image)
    image = table_network.vizaulizate(image)

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        nparr = np.fromstring(bytes_data, np.uint8)
        img_np = cv.imdecode(nparr, cv.IMREAD_COLOR)
        img_np = cv.cvtColor(img_np, cv.COLOR_BGR2RGB)# cv2.IMREAD_COLOR
        table_network.get_network_result(img_np)
        image = table_network.vizaulizate(img_np)

        st.image(image, caption='Sunrise by the mountains',
                 use_column_width=True)

    else:
        st.image(image, caption='Sunrise by the mountains',
                 use_column_width=True)

def page_video():

    # to_show = cv.cvtColor(to_show, cv.COLOR_BGR2RGB)
    # st.image(to_show, caption='Video',
    #          use_column_width=True)

    # f = st.file_uploader("Upload file")
    # if f is not None:
    #     pass
    #     tfile = tempfile.NamedTemporaryFile(delete=False)
    #     tfile.write(f.read())
    #     cap = cv.VideoCapture(tfile.name)
    # else:
    #     cap = cv.VideoCapture('123.mp4')
    # fourcc = cv.VideoWriter_fourcc(*'H264')
    # if cap.isOpened() == False:
    #     st.write("Error opening video stream or file")
    # success, frame = cap.read()
    # if success:
    #     frame_rate = cap.get(cv.CAP_PROP_FPS)
    #     out = cv.VideoWriter('output.mp4', fourcc, frame_rate, (frame.shape[1], frame.shape[0]))
    # while (cap.isOpened()):
    #     success, frame = cap.read()
    #     if success:
    #         table_network.get_network_result(frame)
    #         table_network.get_network_result(frame)
    #         image = table_network.vizaulizate(frame)
    #         out.write(image)
    #     else:
    #         break
    # cap.release()
    # out.release()
    video_file = open('output.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    # st.write(f.shape)

def page_online():
    html = """
                    <h1 align="center"> TrueScan </h1>
         <h4 align="center">Examples of different model predictions</h2>
    """
    st.markdown(html, unsafe_allow_html=True)
    with st.beta_container():
        id = st.slider('Pick diff img', 1, 6, 4)
        st.write(id)
        image = cv.imread('images/{}.png'.format(id))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        table_network.get_network_result(image)
        image = table_network.vizaulizate(image)
        st.image(image, caption='Sunrise by the mountains',
                 use_column_width=True)
        # telegram_bot_sendtext('322')
        # df = pd.DataFrame(np.array([[1, 2, 3,0]]),
        #                    columns=['Radiant_kill', 'Dire_Kill', 'Radiant_tower', 'Dire_Towers'])
        # st.dataframe(df)

def run():
    page_online()
    page_picture()


if __name__ == "__main__":
    run()


