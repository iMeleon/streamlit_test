import cv2 as cv
import numpy as np
import streamlit as st
# from PIL import Image
# import pandas as pd
from collections import OrderedDict
import requests
from darknet import DarkNetwork




#
dota_network = DarkNetwork('models/darknet/yolo_tiny_monitor.cfg',
                            'models/darknet/yolo_tiny_monitor_last.weights',
                            'models/darknet/classes.names',
                           probability_minimum=0.7)
# dota_network = DarkNetwork('models/darknet/yolov4-custom_monitor.cfg',
#                             'models/darknet/yolov4-custom_monitor_last.weights',
#                             'models/darknet/classes.names',
#                            probability_minimum=0.7)
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
    with st.beta_container():
        html_upload = """
             <h2 align="center">Upload picture to predict</h2>
        """
        st.markdown(html_upload, unsafe_allow_html=True)
        image = cv.imread('images/1.png'.format())
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        dota_network.get_network_result(image)
        image = dota_network.vizaulizate(image)
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            bytes_data = uploaded_file.read()
            nparr = np.fromstring(bytes_data, np.uint8)
            img_np = cv.imdecode(nparr, cv.IMREAD_COLOR)
            img_np = cv.cvtColor(img_np, cv.COLOR_BGR2RGB)
            dota_network.get_network_result(img_np)
            image = dota_network.vizaulizate(img_np)

            st.image(image, use_column_width=True)

        else:
            st.image(image, use_column_width=True)


# def page_video():
#
#     # to_show = cv.cvtColor(to_show, cv.COLOR_BGR2RGB)
#     # st.image(to_show, caption='Video',
#     #          use_column_width=True)
#
#
#     cap = cv.VideoCapture('222.mp4')
#     f = st.file_uploader("Upload file")
#     if f is not None:
#         pass
#         # tfile = tempfile.NamedTemporaryFile(delete=False)
#         # tfile.write(f.read())
#         # cap = cv.VideoCapture(tfile.name)
#     else:
#         cap = cv.VideoCapture('222.mp4')
#     fourcc = cv.VideoWriter_fourcc(*'mp4v')
#     if cap.isOpened() == False:
#         st.write("Error opening video stream or file")
#     success, frame = cap.read()
#     if success:
#         frame_rate = cap.get(cv.CAP_PROP_FPS)
#         out = cv.VideoWriter('output.mp4', fourcc, frame_rate, (frame.shape[1], frame.shape[0]))
#     while (cap.isOpened()):
#         success, frame = cap.read()
#         if success:
#
#             dota_network.get_network_result(frame)
#             image = dota_network.vizaulizate(frame)
#             st.write(type(image))
#             st.write(image.shape)
#             st.image(image)
#             out.write(image)
#
#         else:
#             break
#     cap.release()
#     out.release()
#     video_file = open('output.mp4', 'rb')
#     video_bytes = video_file.read()
#     st.video(video_bytes)
#     st.write("Done")

def page_online():
    html = """
                    <h1 align="center"> TrueScan </h1>
         <h4 align="center">Examples of different model predictions</h2>
    """
    st.markdown(html, unsafe_allow_html=True)
    with st.beta_container():
        id_ = st.slider('Pick diff img', 1, 6, 3)
        image = cv.imread('images/{}.png'.format(id_))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        dota_network.get_network_result(image)
        image = dota_network.vizaulizate(image)
        st.image(image, use_column_width=True)
        if dota_network.right_kill: telegram_bot_sendtext('Dire kills:{}'.format(dota_network.right_kill))
        if dota_network.left_kill: telegram_bot_sendtext('Radiant kills:{}'.format(dota_network.left_kill))
        # telegram_bot_sendtext('322')
        # df = pd.DataFrame(np.array([[1, 2, 3,0]]),
        #                    columns=['Radiant_kill', 'Dire_Kill', 'Radiant_tower', 'Dire_Towers'])
        # st.dataframe(df)
        video_file = open('322.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

DEMOS = OrderedDict(
    [("Online", (page_online, None)),
        ("Picture", (page_picture, None)),

    ]
)
def run():
    demo_name = st.sidebar.selectbox("Choose a demo", list(DEMOS.keys()), 0)
    demo = DEMOS[demo_name][0]
    for i in range(10):
        st.empty()

    demo()


if __name__ == "__main__":
    run()


