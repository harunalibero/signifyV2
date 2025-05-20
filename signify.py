import mediapipe as mp
import numpy as np
import cv2
import time
import pyautogui
from enum import Enum

class Gesture(Enum):
    NONE = 0
    SWIPE_LEFT = 1
    SWIPE_RIGHT = 2
    SWIPE_UP = 3
    SWIPE_DOWN = 4
    PINCH = 5
    FIST = 6
    OPEN_PALM = 7
    POINTING = 8
    THUMBS_UP = 9
    OK = 10
    PEACE_SIGN = 11
    STOP = 12

class GestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        self.drawing_utils = mp.solutions.drawing_utils
        self.prev_time = time.time()
        self.fps = 0
        self.velocity_threshold = 2
        self.screen_width, self.screen_height = pyautogui.size()
        self.mouse_positions = []
        self.smoothing_window = 3
        self.current_gesture = Gesture.NONE
        self.prev_landmarks = None
        self.control_mode = "mouse"
        self.last_hand_landmarks = None

    def calculate_fps(self):
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        return fps

    def process_frame(self, frame):
        self.fps = self.calculate_fps()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        output_frame = frame.copy()
        self.current_gesture = Gesture.NONE
        self.last_hand_landmarks = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.last_hand_landmarks = hand_landmarks
            current_landmarks = self._get_normalized_landmarks(hand_landmarks)
            self.current_gesture = self._detect_gesture(current_landmarks)
            self._execute_control(current_landmarks)
            self.prev_landmarks = current_landmarks
            self.drawing_utils.draw_landmarks(
                output_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        self._draw_debug_info(output_frame)
        return output_frame, self.current_gesture

    def _get_normalized_landmarks(self, hand_landmarks):
        lm = hand_landmarks.landmark
        return {
            'wrist': np.array([lm[0].x, lm[0].y]),
            'thumb_tip': np.array([lm[4].x, lm[4].y]),
            'index_tip': np.array([lm[8].x, lm[8].y]),
            'middle_tip': np.array([lm[12].x, lm[12].y]),
            'ring_tip': np.array([lm[16].x, lm[16].y]),
            'pinky_tip': np.array([lm[20].x, lm[20].y])
        }

    def _detect_gesture(self, current_landmarks):
        if self.prev_landmarks is None:
            return Gesture.NONE
        dx = current_landmarks['wrist'][0] - self.prev_landmarks['wrist'][0]
        dy = current_landmarks['wrist'][1] - self.prev_landmarks['wrist'][1]
        velocity = np.sqrt(dx ** 2 + dy ** 2)

        if velocity > self.velocity_threshold:
            if abs(dx) > abs(dy):
                return Gesture.SWIPE_RIGHT if dx > 0 else Gesture.SWIPE_LEFT
            return Gesture.SWIPE_DOWN if dy > 0 else Gesture.SWIPE_UP

        if self._is_pinch(current_landmarks):
            return Gesture.PINCH
        if self._is_fist(current_landmarks):
            return Gesture.FIST
        if self._is_open_palm(current_landmarks):
            return Gesture.OPEN_PALM
        if self._is_pointing(current_landmarks):
            return Gesture.POINTING
        if self._is_thumbs_up(current_landmarks):
            return Gesture.THUMBS_UP
        if self._is_ok(current_landmarks):
            return Gesture.OK
        if self._is_peace_sign(current_landmarks):
            return Gesture.PEACE_SIGN
        if self._is_stop(current_landmarks):
            return Gesture.STOP

        return Gesture.NONE

    def _is_fist(self, landmarks):
        tips = [landmarks[f'{f}_tip'] for f in ['thumb', 'index', 'middle', 'ring', 'pinky']]
        wrist = landmarks['wrist']
        avg_dist = np.mean([np.linalg.norm(tip - wrist) for tip in tips])
        return avg_dist < 0.15

    def _is_open_palm(self, landmarks):
        tips = [landmarks[f'{f}_tip'] for f in ['thumb', 'index', 'middle', 'ring', 'pinky']]
        wrist = landmarks['wrist']
        avg_dist = np.mean([np.linalg.norm(tip - wrist) for tip in tips])
        return avg_dist > 0.25

    def _is_pointing(self, landmarks):
        index_tip = landmarks['index_tip']
        wrist = landmarks['wrist']
        extended = np.linalg.norm(index_tip - wrist) > 0.25
        others = [landmarks[f'{f}_tip'] for f in ['thumb', 'middle', 'ring', 'pinky']]
        closed = all(np.linalg.norm(tip - wrist) < 0.15 for tip in others)
        return extended and closed

    def _is_thumbs_up(self, landmarks):
        return (landmarks['thumb_tip'][1] < landmarks['index_tip'][1] and
                landmarks['thumb_tip'][0] < landmarks['index_tip'][0] and
                np.linalg.norm(landmarks['index_tip'] - landmarks['wrist']) > 0.2)

    def _is_ok(self, landmarks):
        thumb_index = np.linalg.norm(landmarks['thumb_tip'] - landmarks['index_tip'])
        thumb_middle = np.linalg.norm(landmarks['thumb_tip'] - landmarks['middle_tip'])
        return thumb_index < 0.05 and thumb_middle < 0.05

    def _is_peace_sign(self, landmarks):
        return (np.linalg.norm(landmarks['index_tip'] - landmarks['wrist']) > 0.25 and
                np.linalg.norm(landmarks['middle_tip'] - landmarks['wrist']) > 0.25)

    def _is_stop(self, landmarks):
        return self._is_open_palm(landmarks) and not self._is_fist(landmarks)

    def _is_pinch(self, landmarks):
        return np.linalg.norm(landmarks['thumb_tip'] - landmarks['index_tip']) < 0.05

    def _execute_control(self, landmarks):
        if self.current_gesture == Gesture.NONE:
            return
        if self.control_mode == "mouse":
            self._control_mouse(landmarks)
        elif self.control_mode == "keyboard":
            self._control_keyboard()
        elif self.control_mode == "media":
            self._control_media()

        if self.current_gesture in {Gesture.OK, Gesture.THUMBS_UP, Gesture.PEACE_SIGN, Gesture.STOP}:
            print(f"{self.current_gesture.name.replace('_', ' ').title()} Detected")

    def _control_mouse(self, landmarks):
        screen_x = int(landmarks['index_tip'][0] * self.screen_width)
        screen_y = int(landmarks['index_tip'][1] * self.screen_height)
        self.mouse_positions.append((screen_x, screen_y))
        if len(self.mouse_positions) > self.smoothing_window:
            self.mouse_positions.pop(0)
        avg_x = int(np.mean([pos[0] for pos in self.mouse_positions]))
        avg_y = int(np.mean([pos[1] for pos in self.mouse_positions]))
        pyautogui.moveTo(avg_x, avg_y, _pause=False)
        if self.current_gesture == Gesture.PINCH:
            pyautogui.click(_pause=False)
        elif self.current_gesture == Gesture.FIST:
            pyautogui.rightClick(_pause=False)

    def _control_keyboard(self):
        key_map = {
            Gesture.SWIPE_LEFT: 'left',
            Gesture.SWIPE_RIGHT: 'right',
            Gesture.SWIPE_UP: 'up',
            Gesture.SWIPE_DOWN: 'down'
        }
        if self.current_gesture in key_map:
            pyautogui.press(key_map[self.current_gesture], _pause=False)

    def _control_media(self):
        if self.current_gesture == Gesture.OPEN_PALM:
            pyautogui.press('playpause', _pause=False)
        elif self.current_gesture == Gesture.SWIPE_RIGHT:
            pyautogui.press('nexttrack', _pause=False)
        elif self.current_gesture == Gesture.SWIPE_LEFT:
            pyautogui.press('prevtrack', _pause=False)

    def _draw_debug_info(self, frame):
        cv2.putText(frame, f"FPS: {int(self.fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Gesture: {self.current_gesture.name}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Mode: {self.control_mode}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def detect_letter(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    mcp = [2, 5, 9, 13, 17]
    extended = [hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[mcp[i]].y for i in range(5)]
    letters = {
        (False, False, False, False, False): "A",
        (True, True, True, True, True): "B",
        (True, True, False, False, False): "C",
        (True, False, False, False, False): "D",
        (True, True, True, True, True): "E"
    }
    return letters.get(tuple(extended), None)

def main():
    controller = GestureController()

    # Try multiple camera indices if index 0 fails
    for cam_index in range(3):
        cap = cv2.VideoCapture(cam_index)
        if cap.isOpened():
            print(f"[INFO] Camera opened with index {cam_index}")
            break
        else:
            print(f"[WARNING] Could not open camera with index {cam_index}")
            cap.release()
            cap = None

    if cap is None or not cap.isOpened():
        print("[ERROR] Could not open any available video capture device.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 660)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame read failed, exiting loop.")
            break

        frame = cv2.flip(frame, 1)
        processed_frame, _ = controller.process_frame(frame)

        if controller.last_hand_landmarks:
            letter = detect_letter(controller.last_hand_landmarks)
            if letter:
                cv2.putText(processed_frame, f"Letter: {letter}", (50, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow('Gesture Control', processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            controller.control_mode = "mouse"
            print("[MODE] Mouse Control")
        elif key == ord('k'):
            controller.control_mode = "keyboard"
            print("[MODE] Keyboard Control")
        elif key == ord('a'):
            controller.control_mode = "media"
            print("[MODE] Media Control")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
