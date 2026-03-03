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
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils


def reconhecer_mao():

    clickLeft = 0
    clickRight = 0
    isClickingLeft = False
    isClickingRight = False

    while True:
        ret, img = video.read()
        if not ret:
            break

        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        h, w, _ = img.shape

        left_display = None
        right_display = None

        if results.multi_hand_landmarks:
            for idx, handLms in enumerate(results.multi_hand_landmarks):

                hand_label = results.multi_handedness[idx].classification[0].label

                if hand_label == 'Left':

                    distancia, angle, isClickingLeft, clickBinary = draw_hand_info(
                        img, handLms, w, h, isClickingLeft
                    )

                    clickLeft += clickBinary
                    left_display = (distancia, angle, isClickingLeft, clickLeft)

                else:

                    distancia, angle, isClickingRight, clickBinary = draw_hand_info(
                        img, handLms, w, h, isClickingRight
                    )

                    clickRight += clickBinary
                    right_display = (distancia, angle, isClickingRight, clickRight)

        # ================= DISPLAY ESQUERDA =================
        if left_display:

            draw_info_box(
                img,
                position="left",
                width=w,
                data=left_display,
                label="Left"
            )

        # ================= DISPLAY DIREITA =================
        if right_display:

            draw_info_box(
                img,
                position="right",
                width=w,
                data=right_display,
                label="Right"
            )

        cv2.imshow("Hand Recognizer", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    video.release()
    cv2.destroyAllWindows()


def draw_hand_info(img, hand_landmarks, w, h, isClicking):

    thumb = hand_landmarks.landmark[4]
    indexFinger = hand_landmarks.landmark[8]

    x0, y0 = int(thumb.x * w), int(thumb.y * h)
    x1, y1 = int(indexFinger.x * w), int(indexFinger.y * h)

    # Linha e círculos
    cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.circle(img, (x0, y0), 8, (0, 0, 255), -1)
    cv2.circle(img, (x1, y1), 8, (255, 0, 0), -1)

    distancia = int(math.hypot(x1 - x0, y1 - y0))

    clickAtual = distancia < 30
    clickBinary = 0

    # Detecta transição (aberto -> fechado)
    if clickAtual and not isClicking:
        clickBinary = 1

    isClicking = clickAtual

    angle, _ = angle_from_thumb(thumb, indexFinger, w, h)

    mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

    return distancia, angle, isClicking, clickBinary


def draw_info_box(img, position, width, data, label):

    distancia, angle, isClicking, clicks = data

    box_width = 420
    box_height = 160
    margin = 10

    if position == "left":
        x1 = margin
    else:
        x1 = width - box_width - margin

    y1 = margin
    x2 = x1 + box_width
    y2 = y1 + box_height

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

    cv2.putText(img, f"{label} Distancia: {distancia}px",
                (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(img, f"{label} Angulo: {angle} deg",
                (x1 + 10, y1 + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(img, f"Is Clicking: {isClicking}",
                (x1 + 10, y1 + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(img, f"Clicks: {clicks}",
                (x1 + 10, y1 + 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def angle_from_thumb(thumb, index, w, h):

    x0, y0 = int(thumb.x * w), int(thumb.y * h)
    x1, y1 = int(index.x * w), int(index.y * h)

    dx = x1 - x0
    dy = y0 - y1  # eixo Y invertido

    angle = math.degrees(math.atan2(dy, dx))

    if angle < 0:
        angle += 360

    return int(angle), (x0, y0)


# Executar
reconhecer_mao()