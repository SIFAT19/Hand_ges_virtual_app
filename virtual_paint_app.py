import cv2
import mediapipe as mp
import numpy as np
import time
import math

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize MediaPipe hands module
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

# Create two canvases
camera_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)  # Camera + drawings
drawing_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)  # Only drawings
drawing_canvas[:] = (255, 255, 255)  # White background for drawing canvas
camera_canvas[:] = (60, 60, 60)  # Darker background 

prev_x, prev_y = 0, 0
painting = False
brush_color = (0, 0, 255)  
brush_size = 10
eraser_size = 25 
smoothing_factor = 0.5

# More  colors (BGR format)
colors = [
    (0, 0, 220),     # Stronger Red
    (0, 220, 0),     # Stronger Green
    (220, 0, 0),     # Stronger Blue
    (0, 220, 220),   # Stronger Yellow
    (220, 0, 220),   # Stronger Magenta
    (220, 220, 220)  # Brighter White
]

pTime = 0
button_area_top = 50
button_area_bottom = 120
button_margin = 20

def draw_color_buttons(img, active_color):
    h, w = img.shape[:2]
    button_size = 60
    total_width = len(colors) * (button_size + button_margin) - button_margin
    start_x = (w - total_width) // 2
    
    # Draw solid, dark background for buttons
    cv2.rectangle(img, (0, button_area_top - 10), (w, button_area_bottom + 10), (40, 40, 40), -1)
    
    # Draw color selection buttons with brighter colors
    for i, color in enumerate(colors):
        x1 = start_x + i * (button_size + button_margin)
        x2 = x1 + button_size
        y1 = button_area_top
        y2 = button_area_bottom
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        
        if color == active_color:
            cv2.rectangle(img, (x1-3, y1-3), (x2+3, y2+3), (255, 255, 255), 3)
    
    return img

def check_color_selection(x, y, img):
    button_size = 60
    total_width = len(colors) * (button_size + button_margin) - button_margin
    start_x = (img.shape[1] - total_width) // 2
    
    if button_area_top <= y <= button_area_bottom:
        for i, color in enumerate(colors):
            x1 = start_x + i * (button_size + button_margin)
            x2 = x1 + button_size
            
            if x1 <= x <= x2:
                return color
    return None

def is_eraser_active(hand_landmarks, img_shape, threshold=0.07):
    h, w = img_shape[:2]
    
    # Get index (8) and middle (12) finger tips
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    
    # Calculate absolute distance in pixels
    index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
    middle_x, middle_y = int(middle_tip.x * w), int(middle_tip.y * h)
    distance = math.sqrt((index_x - middle_x)**2 + (index_y - middle_y)**2)
    
    # Check if fingers are close AND middle finger is somewhat raised
    middle_raised = middle_tip.y < hand_landmarks.landmark[9].y
    return distance < (w * threshold) and middle_raised

def is_index_up(hand_landmarks):
    # Check if index finger is up and middle is down
    index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y - 0.05
    middle_down = hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y
    return index_up and middle_down

# Create windows
cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
cv2.namedWindow("Drawing Only", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera View", 1280, 720)
cv2.resizeWindow("Drawing Only", 1280, 720)

while True:
    success, img = cap.read()
    if not success:
        break

    # Flip and resize the image
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (1280, 720))
    h, w, _ = img.shape
    
    # Reset camera canvas with darker background
    camera_canvas = img.copy()
    camera_canvas = cv2.addWeighted(camera_canvas, 0.1, np.zeros_like(camera_canvas), 0.3, 0)  # Darken camera feed
    camera_canvas = draw_color_buttons(camera_canvas, brush_color)
    
    # Process hand landmarks
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB) 
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Get landmarks
            index_tip = handLms.landmark[8]
            index_x = int(index_tip.x * w)
            index_y = int(index_tip.y * h)
            
            # Apply smoothing
            if painting:
                index_x = int(prev_x * (1 - smoothing_factor) + index_x * smoothing_factor)
                index_y = int(prev_y * (1 - smoothing_factor) + index_y * smoothing_factor)
            
            # Check eraser mode
            eraser_mode = is_eraser_active(handLms, img.shape)
            
            # Draw appropriate cursor on camera view
            cursor_color = (255, 255, 255) if not eraser_mode else (0, 0, 0)
            cursor_size = eraser_size if eraser_mode else brush_size
            cv2.circle(camera_canvas, (index_x, index_y), cursor_size//2 + 3, cursor_color, 2)
            
            # Check if in drawing mode
            if is_index_up(handLms) or eraser_mode:
                # Check if we're in button area
                if button_area_top <= index_y <= button_area_bottom and not eraser_mode:
                    selected_color = check_color_selection(index_x, index_y, camera_canvas)
                    if selected_color is not None:
                        brush_color = selected_color
                        # Visual confirmation
                        cv2.circle(camera_canvas, (index_x, index_y), 30, (255, 255, 255), 2)
                else:
                    # In drawing/erasing area
                    if not painting:
                        prev_x, prev_y = index_x, index_y
                        painting = True
                    
                    # Draw or erase based on mode (on drawing canvas)
                    if eraser_mode:
                        cv2.line(drawing_canvas, (prev_x, prev_y), (index_x, index_y), (255, 255, 255), eraser_size)
                    else:
                        cv2.line(drawing_canvas, (prev_x, prev_y), (index_x, index_y), brush_color, brush_size)
                    
                    prev_x, prev_y = index_x, index_y
            else:
                painting = False
            
            # Draw hand landmarks on camera view
            mpDraw.draw_landmarks(camera_canvas, handLms, mpHands.HAND_CONNECTIONS)
    
    # Combine camera with drawings using better blending
    camera_view = cv2.addWeighted(camera_canvas, 0.6, drawing_canvas, 0.7, 0)
    
    # Display FPS on camera view
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(camera_view, f"FPS: {int(fps)}", (w-150, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show instructions on camera view with better contrast
    cv2.putText(camera_view, "Index up to draw", (w-200, 170), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(camera_view, "Index+Middle close = Eraser", (w-350, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(camera_view, "Point at colors to select", (w-350, 230), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(camera_view, "'c' to clear", (w-150, 260), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add label to drawing window
    drawing_display = drawing_canvas.copy()
    cv2.putText(drawing_display, "Your Drawing (White Background)", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Show both windows
    cv2.imshow("Camera View", camera_view)
    cv2.imshow("Drawing Only", drawing_display)
    
    key = cv2.waitKey(1)
    if key == ord('c'):
        drawing_canvas[:] = (255, 255, 255)  # Clear drawing canvas to white
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
