import cv2 as cv
import numpy as np
import os
from time import time, sleep
import streamlit as st

class DarkNetwork:

    def __init__(self, cfg_file_name, weights_file_name, class_file_name, probability_minimum=0.8, threshold=0.3):
        with open(class_file_name) as f:
            self.labels = [line.strip() for line in f]
        self.network = cv.dnn.readNetFromDarknet(cfg_file_name, weights_file_name)
        self.layers_names_all = self.network.getLayerNames()
        self.layers_names_output = [self.layers_names_all[i[0] - 1] for i in self.network.getUnconnectedOutLayers()]
        self.probability_minimum = probability_minimum
        self.threshold = threshold
        self.class_colours = np.random.randint(0, 255, size=(len(self.labels), 3), dtype='uint8')
        self.bounding_boxes = []
        self.confidences = []
        self.class_numbers = []
        self.detected_objects = []
        self.indices_after_NMSBoxes = None
        self.save_image_counter = 155
        self.objects_after_NMS = {}

    def get_network_result(self, photo):
        if 0 in photo.shape:
            print('Error in shape:', photo.shape)
            return None
        self.bounding_boxes = []
        self.confidences = []
        self.class_numbers = []
        self.detected_objects = []
        self.indices_after_NMSBoxes = []
        self.objects_after_NMS = {}
        h, w = photo.shape[:2]
        blob = cv.dnn.blobFromImage(photo, 1 / 255.0, (416, 416),
                                    swapRB=False, crop=False)
        self.network.setInput(blob)  # setting blob as input to the network
        # start = time()
        output_from_network = self.network.forward(self.layers_names_output)
        #print(len(output_from_network))
        # end = time()
        # print('Objects Detection took {:.5f} seconds'.format(end - start))
        # Preparing lists for detected bounding boxes,
        # obtained confidences and class's number
        # print(output_from_network)
        # Going through all output layers after feed forward pass
        for result in output_from_network:
            # Going through all detections from current output layer
            for detected_objects in result:
                # Getting 80 classes' probabilities for current detected object
                scores = detected_objects[5:]
                # Getting index of the class with the maximum value of probability
                class_current = np.argmax(scores)

                # Getting value of probability for defined class
                confidence_current = scores[class_current]

                # # Check point
                # # Every 'detected_objects' numpy array has first 4 numbers with
                # # bounding box coordinates and rest 80 with probabilities for every class
                # print(detected_objects.shape)  # (85,)

                # Eliminating weak predictions with minimum probability
                # print(scores)
                if confidence_current > self.probability_minimum:
                    # print(scores)
                    # print(class_current)
                    # Scaling bounding box coordinates to the initial image size
                    # YOLO data format keeps coordinates for center of bounding box
                    # and its current width and height
                    # That is why we can just multiply them elementwise
                    # to the width and height
                    # of the original image and in this way get coordinates for center
                    # of bounding box, its width and height for original image
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])
                    # Now, from YOLO data format, we can get top left corner coordinates
                    # that are x_min and y_min
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))
                    if x_min < 0:
                        x_min = 0
                    if y_min < 0:
                        y_min = 0
                    # Adding results into prepared lists
                    self.bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    self.confidences.append(round(float(confidence_current), 3))
                    self.class_numbers.append(class_current)
                    self.detected_objects.append(
                        photo[y_min:y_min + int(box_height)+2, x_min:x_min + int(box_width)])
        # self.bounding_boxes =  self.bounding_boxes[0]
        # print(self.bounding_boxes)
        self.indices_after_NMSBoxes = cv.dnn.NMSBoxes(self.bounding_boxes, self.confidences,
                                                      self.probability_minimum,
                                                      self.threshold)
        # self.indices_after_NMSBoxes = np.array([[x] for x in range(len(self.class_numbers))])
        # print(self.indices_after_NMSBoxes)
        if len(self.indices_after_NMSBoxes) > 0:
            for i in self.indices_after_NMSBoxes.flatten():
                obj_photo = self.detected_objects[i]
                class_name = self.labels[int(self.class_numbers[i])]
                if class_name not in self.objects_after_NMS:
                    self.objects_after_NMS[class_name] = []
                self.objects_after_NMS[class_name].append(obj_photo)
        return 0

    def save_detected_obj(self, path, id_class=0):
        if len(self.indices_after_NMSBoxes) > 0:
            # Going through indexes of results
            for i in self.indices_after_NMSBoxes.flatten():
                if self.class_numbers[i] == id_class:
                    directory = path
                    os.chdir(directory)
                    print("Before saving image:")
                    print(os.listdir(directory))
                    filename = '{}.png'.format(self.save_image_counter)
                    self.save_image_counter += 1
                    cv.imwrite(filename, self.detected_objects[i])

    def get_detected_obj(self, id_class=0):
        result = []
        if len(self.indices_after_NMSBoxes) > 0:

            # Going through indexes of results
            for i in self.indices_after_NMSBoxes.flatten():
                if self.class_numbers[i] == id_class:
                    result.append(self.detected_objects[i])

        return result

    def vizaulizate(self, photo):
        photo = photo.copy()
        counter = 0
        # print('Total objects been detected before NMS:', len(self.bounding_boxes))
        # Checking if there is at least one detected object after non-maximum suppression
        if len(self.indices_after_NMSBoxes) > 0:
            # Going through indexes of results
            for i in self.indices_after_NMSBoxes.flatten():
                # Showing labels of the detected objects
                # print(self.class_numbers)
                # print(i)
                # print(int(self.class_numbers[i]))
                # print('Object {0}: {1}'.format(counter, self.labels[int(self.class_numbers[i])]))

                # Incrementing counter
                counter += 1

                # Getting current bounding box coordinates,
                # its width and height
                x_min, y_min = self.bounding_boxes[i][0], self.bounding_boxes[i][1]
                box_width, box_height = self.bounding_boxes[i][2], self.bounding_boxes[i][3]

                # Preparing colour for current bounding box
                # and converting from numpy array to list
                # print('loh', self.class_numbers)
                # print(self.class_numbers[i])
                colour_box_current = self.class_colours[self.class_numbers[i]].tolist()

                # # # Check point
                # print(type(colour_box_current))  # <class 'list'>
                # print(colour_box_current)  # [172 , 10, 127]

                # Drawing bounding box on the original image
                cv.rectangle(photo, (x_min, y_min),
                             (x_min + box_width, y_min + box_height),
                             colour_box_current, 2)

                # Preparing text with label and confidence for current bounding box
                text_box_current = '{}: {:.4f}'.format(self.labels[int(self.class_numbers[i])],
                                                       self.confidences[i])

                # Putting text with label and confidence on the original image
                cv.putText(photo, text_box_current, (x_min, y_min - 5),
                           cv.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)
            # print('Number of objects left after non-maximum suppression:', counter)

        return photo



if __name__ == "__main__":


    table_network = DarkNetwork('models/table/wm_tables.cfg',
                                'models/table/wm_tables_last.weights',
                                'models/table/classes2.names',
                                probability_minimum=0.7)
    # wm_table = Table('models/wm_tables_only/yolov4_wm_tables_only.cfg',
    #                  'models/wm_tables_only/yolov4_wm_tables_only_last.weights',
    #                  'models/wm_tables_only/classes.names',
    #                  probability_minimum=0.2)

image = cv.imread('58.png')
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
table_network.get_network_result(image)
image = table_network.vizaulizate(image)
st.write("""
# My first app
Vis3on22[8] *predictor!*
""")
st.image(image, caption='Sunrise by the mountains',
         use_column_width=True)
print(228)
#cv.imshow('table2', screenshot_with_bb)
        # table_network.save_detected_obj(r'C:\Users\user\Desktop\test',0)
        # if 'preflop' in table_network.objects_after_NMS:
        #     # table = table_network.objects_after_NMS['preflop'][0]
        #     # print(type(table_network.objects_after_NMS['preflop']))
        #     for table in table_network.objects_after_NMS['preflop']:
        #         wm_table.get_network_result(table, False)
        #         # for k, v in wm_table.table_items.items():
                #     print(k, v)
                # cv.imshow('left', wm_table.visual_dict['right'])
                # cv.imshow('right', wm_table.visual_dict['right'])
            # visual_dict['right']
            #sleep(2)

