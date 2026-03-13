import subprocess
import os
import sys
import glob

MOVEMENT_SCRIPT = "calibrationMovementV2.py"
ANALYSIS_SCRIPT = "calculateTransformationMatrixV2.py"

def get_latest_calibration_folder():
    """Sucht den zuletzt erstellten calibration_data_ Ordner."""
    folders = glob.glob("calibration_data_*")
    if not folders:
        return None
    return max(folders, key=os.path.getmtime)

def main():
    print("=== Starte Kalibrierungsprozess ===")
    
    print(f"\n[Schritt 1] Führe {MOVEMENT_SCRIPT} aus...")
    if not os.path.exists(MOVEMENT_SCRIPT):
        print(f"Fehler: {MOVEMENT_SCRIPT} wurde im aktuellen Verzeichnis nicht gefunden.")
        sys.exit(1)
        
    result_movement = subprocess.run([sys.executable, MOVEMENT_SCRIPT])
    
    if result_movement.returncode != 0:
        print(f"\nFehler: Die Datenerfassung ({MOVEMENT_SCRIPT}) ist abgestürzt. Breche Kalibrierung ab.")
        sys.exit(1)

    latest_folder = get_latest_calibration_folder()
    if not latest_folder:
        print("\nFehler: Es konnte kein neu erstellter 'calibration_data_...'-Ordner gefunden werden.")
        sys.exit(1)
        
    print(f"\nDaten erfolgreich erfasst in: {latest_folder}")

    print(f"\n[Schritt 2] Führe {ANALYSIS_SCRIPT} für den Ordner {latest_folder} aus...")
    if not os.path.exists(ANALYSIS_SCRIPT):
        print(f"Fehler: {ANALYSIS_SCRIPT} wurde im aktuellen Verzeichnis nicht gefunden.")
        sys.exit(1)
        
    result_analysis = subprocess.run([sys.executable, ANALYSIS_SCRIPT, latest_folder])
    
    if result_analysis.returncode != 0:
        print(f"\nFehler: Die Auswertung ({ANALYSIS_SCRIPT}) wurde mit einem Fehler beendet.")
        sys.exit(1)
        
    print("\n=== Kalibrierungsprozess vollständig abgeschlossen ===")

if __name__ == "__main__":
    main()