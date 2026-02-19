import cv2

def check_cameras(max_index=10):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Index {i} ist verfügbar. Öffne Fenster...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                cv2.imshow(f"Kamera Index {i} - Beenden mit 'q'", frame)
                
                # Warte auf Taste 'q' zum Schließen dieses Fensters
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        else:
            print(f"Index {i} nicht verfügbar.")

if __name__ == "__main__":
    check_cameras()