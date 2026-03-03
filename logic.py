import cv2
import mediapipe as mp
import math


###########################
vw, vh = 1200, 480
###########################

video = cv2.VideoCapture(0)
video.set(3, vw)
video.set(4, vh)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

def reconhecer_mao():
    while True:
        ret, img = video.read()
        if not ret:
            break

        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        h, w, _ = img.shape

        left_display = (0, 0)
        right_display = (0, 0)

        if results.multi_hand_landmarks:
            for idx, handLms in enumerate(results.multi_hand_landmarks):
                distancia, angle, isClicking = draw_hand_info(img, handLms, w, h)
                
                # Detectar se é mão esquerda ou direita
                hand_label = results.multi_handedness[idx].classification[0].label
                
                if hand_label == 'Left':
                    left_display = (distancia, angle, isClicking)
                else:
                    right_display = (distancia, angle, isClicking)

        # Mostrar valores no canto superior esquerdo (mão esquerda)
        if left_display != (0, 0):
            cv2.rectangle(img, (10, 10), (600, 100), (0, 0, 0), -1)
            cv2.putText(img, f"Left Distancia: {left_display[0]}px", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, f"Left Angulo: {left_display[1]} deg", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(img, f"Clicks Left: {left_display[2]}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Mostrar valores no canto superior direito (mão direita)
        if right_display != (0, 0):
            cv2.rectangle(img, (10, 10), (600, 100), (0, 0, 0), -1)
            cv2.putText(img, f"Right Distancia: {right_display[0]}px", (w - 350, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, f"Right Angulo: {right_display[1]} deg", (w - 350, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(img, f"Clicks mão direita: {right_display[2]}", (w - 350, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("imagem", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    video.release()
    cv2.destroyAllWindows()

clickLeft=0
clickRight=0
def draw_hand_info(img, hand_landmarks, w, h):
    thumb = hand_landmarks.landmark[4]
    index = hand_landmarks.landmark[8]
    isClicking = False

    x0, y0 = int(thumb.x * w), int(thumb.y * h)
    x1, y1 = int(index.x * w), int(index.y * h)

    # Linha e círculos
    cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.circle(img, (x0, y0), 8, (0, 0, 255), -1)
    cv2.circle(img, (x1, y1), 8, (255, 0, 0), -1)

    # Distância
    distancia = int(math.hypot(x1 - x0, y1 - y0))

    #Detecção de cliques
    if distancia < 30:
        isClicking = True


    # Ângulo
    angle, _ = angle_from_thumb(thumb, index, w, h)

    # Desenhar landmarks
    mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

    return distancia, angle, isClicking


def angle_from_thumb(thumb, index, w, h):
    x0, y0 = int(thumb.x * w), int(thumb.y * h)
    x1, y1 = int(index.x * w), int(index.y * h)

    dx = x1 - x0
    dy = y0 - y1  # invertido pelo eixo da imagem

    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 360
    return int(angle), (x0, y0)
