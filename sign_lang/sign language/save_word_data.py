import cv2
import os

def save_gesture_data():
    word = input("Enter the word/gesture name (e.g. hello, good, bad): ").lower().strip()
    save_path = os.path.join("dataset", word)
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    print(f"Ready to record '{word}'. Press '\\' to capture, ';' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[50:450, 50:450]
        cv2.rectangle(frame, (50, 50), (450, 450), (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Collecting Data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('\\'):
            img_name = os.path.join(save_path, f"{count}.jpg")
            cv2.imwrite(img_name, roi)
            count += 1
            print(f"Captured {count} images for '{word}'")
        elif key == ord(';'):
            break

    print(f"Data collection for '{word}' completed. {count} images saved.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    save_gesture_data()
