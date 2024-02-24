import copy
import itertools
import json
import csv
import time

import cv2
import mediapipe as mp
import numpy as np
import keyboard
from pynput.keyboard import Key, Controller

from model import modeltrainer

from model.keypoint_classifier import keypoint_classifier as KeyPointClassifier



class Paths:
    def __init__(self, json_path, dataset, model_save_path, tflite_save_path,  model_path):
        self.json_path = json_path
        self.dataset = dataset
        self.model_save_path = model_save_path
        self.tflite_save_path = tflite_save_path
        self.model_path = model_path

class KeySpell:
    def __init__(self, name,gid,func):
        self.name=name
        self.gid =gid
        self.func = func
    def print(self):
        print(self.name,self.gid,self.func)

    def performSpell(self):
        keys = self.func.split('+')

        # Create a new keyboard controller
        keyboard = Controller()

        # Press each key in sequence
        try:
            pressed=[]
            for key in keys:
                try:
                    # Convert the key name string into a Key constant
                    key_constant = getattr(Key, key.split('.')[1])
                    # Press the key
                    keyboard.press(key_constant)
                    pressed.append(key_constant)
                except IndexError:
                    keyboard.press(key)
                    print(key)
                    pressed.append(key)
            # Release each key in reverse order
            for key in reversed(keys):
                try:
                    # Convert the key name string into a Key constant
                    key_constant = getattr(Key, key.split('.')[1])
                    # Release the key
                    keyboard.release(key_constant)
                    pressed.remove(key_constant)
                except IndexError:
                    keyboard.release(key)
                    pressed.remove(key)
                    print(key)

            time.sleep(2)
        except ValueError:
            for key in pressed:
                keyboard.release(key)


    @classmethod
    def parse_json(cls, json_file_path):
        with open(json_file_path) as f:
            data = json.load(f)
            gestures = data['gestures']
            keys_spells = []
            for gesture in gestures:
                keys_spells.append(cls(gesture['name'], gesture['id'], gesture['function']))
            return keys_spells

class KeySpellList:
    def __init__(self, paths):
        self.key_spells = {}
        self.paths = paths
        self.json_path = self.paths.json_path
        self.load_data()
        self.dataset = self.paths.dataset


    def __enter__(self):
        self.key_spells.clear()
        self.load_data()
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        self.save_data()

    def load_data(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)
            self.key_spells.clear()
            for item in data['gestures']:
                key_spell = KeySpell(item['name'], item['gid'], item['func'])
                if key_spell.name in self.key_spells or key_spell.gid in self.key_spells:
                    continue
                if any(existing_key_spell.name == key_spell.name or existing_key_spell.gid == key_spell.gid for
                       existing_key_spell in self.key_spells.values()):
                    continue
                self.key_spells[key_spell.name] = key_spell

    def save_data(self):
        data = {'gestures': []}
        for key_spell in self.key_spells.values():
            data['gestures'].append({
                'name': key_spell.name,
                'gid': key_spell.gid,
                'func': key_spell.func,
            })

        with open(self.json_path, 'w') as f:
            json.dump(data, f)

    def get_key_spells(self):
        return list(self.key_spells.values())

    def get_max_gid(self):
        return max([key_spell.gid for key_spell in self.get_key_spells()])

    def add_key_spell(self, key_spell, replace=False):  #replace True for edit?
        if not replace and (key_spell.name in self.key_spells or key_spell.gid in self.key_spells):
            return False
        for existing_key_spell in self.key_spells.values():
            if existing_key_spell.name == key_spell.name or existing_key_spell.gid == key_spell.gid:
                if replace:
                    self.remove_key_spell(existing_key_spell)
                else:
                    return False
        self.key_spells[key_spell.name] = key_spell
        return True

    def remove_key_spell(self, key_spell):
        if key_spell.name in self.key_spells:
            gid_value = key_spell.gid
            # Remove entries from the CSV file with matching gid value
            flag=False
            with open(self.dataset, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row[0] == str(gid_value):
                        flag = True

            # Delete the key_spell object
            if flag:
                print("trained")
                with open(self.dataset, 'r') as f:
                    reader = csv.reader(f)
                    rows = [row for row in reader if row[0] != str(gid_value)]
                with open(self.dataset, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
                SpellTrainer(self.key_spells, key_spell, self.paths).trainModel(delete=False)
            else:
                print("none")
            del self.key_spells[key_spell.name]

            return True
        return False

class KeySpellListManager:
    def __init__(self, paths):
        self.json_path = self.paths.json_path
        self.paths = self.paths

    def __enter__(self):
        self.key_spell_list = KeySpellList(self.paths)
        #self.key_spell_list.load_data()
        return self.key_spell_list

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"An exception of type {exc_type} occurred: {exc_val}")
            return False
        self.key_spell_list.save_data()



class Spellcaster:
    def __init__(self, paths, cap=0):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.paths = paths
        self.keypoint_classifier = KeyPointClassifier.KeyPointClassifier(model_path=self.paths.model_path)
        self.cap = cap

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        return [[min(int(landmark.x * image_width), image_width - 1),
                 min(int(landmark.y * image_height), image_height - 1)]
                for landmark in landmarks.landmark]

    def pre_process_landmark(self, landmark_list):
        # Convert to relative coordinates
        base_x, base_y = landmark_list[0]
        relative_landmark_list = [[x - base_x, y - base_y]
                                  for x, y in landmark_list]

        # Convert to a one-dimensional list and normalize
        normalized_landmark_list = np.array(relative_landmark_list).flatten() / \
                                   max([abs(x) for x in np.array(relative_landmark_list).flatten()])

        return normalized_landmark_list.tolist()

    def wait_for_spell(self, feed=False):
        with self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
        ) as hands:
            cap = cv2.VideoCapture(self.cap)
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Could not read from camera")
                    break



                # Convert image to RGB and detect hands
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                # Draw hand landmarks on image
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:

                        '''
                        self.mp_drawing.draw_landmarks(
                            image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        '''

                        landmark_list = self.calc_landmark_list(image_rgb, hand_landmarks)

                        # Conversion to relative coordinates / normalized coordinates
                        pre_processed_landmark_list = self.pre_process_landmark(
                            landmark_list)

                        hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                        print(hand_sign_id)
                        with KeySpellList(self.paths) as ksl:
                            for key_spell in ksl.get_key_spells():
                                if key_spell.gid == hand_sign_id:
                                    key_spell.performSpell()


                # Display image
                if feed:
                    cv2.imshow("Spellcaster", image)

                # Wait for key press and exit if 'q' is pressed
                if cv2.waitKey(1) == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()


class SpellTrainer:
    def __init__(self, key_spells, key_spell,
                 paths, cap=0):
        self.paths = paths
        self.gesture_dict = {key_spell.name: key_spell.gid}
        self.gesture_name = key_spell.name
        self.maxid = self.gen_max_gid()
        self.cap= cap
        self.dataset= self.paths.dataset
        self.model_save_path = self.paths.model_save_path
        self.tflite_save_path = self.paths.tflite_save_path


    def  gen_max_gid(self):
        with KeySpellList(self.paths) as keyspell_list:
            maxid = keyspell_list.get_max_gid()
            print("maxid: ", maxid)
        return maxid

    def detect_hands(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands

        cap = cv2.VideoCapture(self.cap)
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:
            while cap.isOpened():

                key = cv2.waitKey(10)

                # read frame from webcam
                ret, frame = cap.read()
                if not ret:
                    print("Unable to capture video")
                    break
                # flip the image horizontally
                frame = cv2.flip(frame, 1)

                # detect hand landmarks
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # draw hand landmarks on the image
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # Landmark calculation
                        landmark_list = self.calc_landmark_list(image, hand_landmarks)

                        # Conversion to relative coordinates / normalized coordinates
                        pre_processed_landmark_list = self.pre_process_landmark(
                            landmark_list)
                        # Write to the dataset file
                        if self.checkHeld():
                            self.logging_csv(1, pre_processed_landmark_list)

                        if self.startTrainer():
                            cap.release()
                            cv2.destroyAllWindows()
                            self.trainModel()
                            print("alt")


                # display the image
                cv2.imshow('Hand Gesture Detection', frame)

                # exit if escape key is pressed
                if cv2.waitKey(10) & 0xFF == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()

    def startTrainer(self):
        return keyboard.is_pressed('alt')

    def checkHeld(self):
        return keyboard.is_pressed('ctrl')

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    def logging_csv(self, mode, landmark_list):
        print("gesture_dict:", self.gesture_dict)
        print("gesture_name", self.gesture_name)
        if mode == 1 and self.gesture_name in self.gesture_dict:
            csv_path = 'model/keypoint_classifier/keypointtest.csv'
            with open(csv_path, 'a', newline="") as f:
                print("TRAINING: ", self.gesture_name)
                writer = csv.writer(f)
                writer.writerow([self.gesture_dict[self.gesture_name], *landmark_list])

    def trainModel(self, delete=False):
        pass
        if not delete:
            num_of_classes = self.maxid+1
            print(self.maxid)

        else:
            num_of_classes = self.maxid - 1
            print(self.maxid)

        modeltrainer.trainModel(num_of_classes, dataset=self.dataset, model_save_path=self.model_save_path,
                                tflite_save_path=self.tflite_save_path)
        print("TRAINING DONE")


'''
paths = Paths(json_path='data.json',
dataset='model/keypoint_classifier/keypointtest.csv',
model_save_path ='model/keypoint_classifier/keypoint_classifiertest.hdf5',
tflite_save_path = 'model/keypoint_classifier/keypoint_classifiertest.tflite',
model_path='model/keypoint_classifier/keypoint_classifiertest.tflite')

star = Spellcaster(paths)
star.wait_for_spell(True)

'''


'''

    
'''

'''
#train
with KeySpellList('data.json') as ksl:
    for key_spell in ksl.get_key_spells():
        if key_spell.name == 'terkuda':
            st = SpellTrainer(ksl.get_key_spells(), key_spell)
            st.detect_hands()

cccc
'''

'''
with KeySpellList(paths) as ksl:
    for key_spell in ksl.get_key_spells():
        if key_spell.name == 'thumbs up':
            ksl.remove_key_spell(key_spell)
'''

'''

#perform task:
with KeySpellList('data.json') as ksl:
    for key_spell in ksl.get_key_spells():
        if key_spell.name == "Pointer":
            key_spell.func = "Key.ctrl+Key.cmd+Key.right"
            key_spell.performSpell()
            
'''

