import subprocess
import os
import glob
import sys

MOVEMENT_SCRIPT = "calibrationMovement.py"
ANALYSIS_SCRIPT = "calculateTransformationMatrix.py"

def get_latest_calibration_folder():
    folders = glob.glob("calibration_data_*")
    if not folders:
        return None
    return max(folders, key=os.path.getmtime)

def main():
    print("=== Starte Kalibrierungsprozess ===")
    
    print(f"\n[Schritt 1] Starte Datenerfassung ({MOVEMENT_SCRIPT})...")
    if not os.path.exists(MOVEMENT_SCRIPT):
        print(f"Fehler: Datei '{MOVEMENT_SCRIPT}' nicht gefunden.")
        sys.exit(1)
        
    result_movement = subprocess.run([sys.executable, MOVEMENT_SCRIPT])
    
    if result_movement.returncode != 0:
        print("\nFehler bei der Datenerfassung (Roboterbewegung). Breche ab.")
        sys.exit(1)

    latest_folder = get_latest_calibration_folder()
    if not latest_folder:
        print("\nFehler: Konnte nach der Bewegung keinen neu erstellten 'calibration_data'-Ordner finden.")
        sys.exit(1)
        
    print(f"\nDaten erfolgreich erfasst in Ordner: {latest_folder}")

    print(f"\n[Schritt 2] Starte Auswertung ({ANALYSIS_SCRIPT})...")
    if not os.path.exists(ANALYSIS_SCRIPT):
        print(f"Fehler: Datei '{ANALYSIS_SCRIPT}' nicht gefunden.")
        sys.exit(1)
        
    result_analysis = subprocess.run([sys.executable, ANALYSIS_SCRIPT, latest_folder])
    
    if result_analysis.returncode != 0:
        print("\nFehler bei der Auswertung der Bilder.")
        sys.exit(1)
        
    print("\n=== Kalibrierungsprozess vollständig abgeschlossen ===")

if __name__ == "__main__":
    main()