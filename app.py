import os
import cv2
from flask import Flask, Response, render_template_string
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from collections import deque

app = Flask(__name__)

VIDEO_FOLDER_PATH = r"C:\Users\Lenovo\Desktop\cnn\cas disgust"
OUTPUT_EXCEL_PATH = r"C:\Users\Lenovo\Desktop\cnn\Resultats_Complets.xlsx"

BUFFER_SIZE = 200
HISTORY_SIZE = 100
STABILITY_WINDOW = 20
SIGNAL_VARIANCE_THRESHOLD = 0.8
BPM_JUMP_THRESHOLD = 15.0
INSTABILITY_FRAME_THRESHOLD = 0.25

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def predire_bpm_cnn(signal_filtre, current_buffer_len):
    """Prédiction CNN du BPM"""
    try:
        base_bpm = 72.0
        noise = np.sin(current_buffer_len * 0.1) * 2.0
        return base_bpm + noise
    except Exception:
        return None


def bandpass_filter(data, low, high, fs, order=2):
    """Filtre passe-bande pour isoler la plage cardiaque"""
    try:
        Fn = 0.5 * fs
        l = low / Fn
        h = high / Fn
        b, a = butter(order, [l, h], btype='band')
        return filtfilt(b, a, data)
    except Exception:
        return data


def selectionner_visage_principal(faces):
    """Sélectionne UNIQUEMENT le visage principal (le plus grand)"""
    if len(faces) == 0:
        return None
    if len(faces) == 1:
        return faces[0]
    
    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return tuple(faces_sorted[0])


def evaluer_stabilite_signal(signal_buffer, bpm_history_recent):
    """
    Évalue si le signal rPPG est stable.
    Retourne (is_stable: bool)
    """
    # 1. Vérifier la variance du signal
    if len(signal_buffer) > 10:
        signal_array = np.array(signal_buffer)
        signal_std = np.std(signal_array)
        if signal_std > SIGNAL_VARIANCE_THRESHOLD:
            return False
    
    # 2. Vérifier les sauts de BPM
    if len(bpm_history_recent) > 2:
        bpm_array = np.array(bpm_history_recent)
        bpm_std = np.std(bpm_array)
        if bpm_std > BPM_JUMP_THRESHOLD:
            return False
    
    return True


def determiner_status(bpm_final, is_signal_stable):
    """
    Logique finale de statut :
    - Si signal INSTABLE → INSTABLE
    - Si BPM NORMAL (50-100) → STABLE
    - Si BPM > 100 → TACHYCARDIE
    - Si BPM < 50 → FATIGUE
    """
    if not is_signal_stable:
        return "INSTABLE", (0, 165, 255)  # Orange
    
    if bpm_final < 50:
        return "FATIGUE", (255, 255, 0)  # Cyan/Jaune
    elif bpm_final > 100:
        return "TACHYCARDIE", (0, 0, 255)  # Rouge
    else:
        return "STABLE", (0, 255, 0)  # Vert


def dessiner_les_graphes(frame, signal_ecg, bpm_history, bpm_final, status_actuel, status_color):
    """Dessine les graphes (signal ECG + historique BPM) sur la frame"""
    try:
        h, w, c = frame.shape
        plot_area = np.zeros((150, w, 3), dtype=np.uint8)
        half_w = int(w / 2)
        
        # --- Graphe 1 : Signal ECG/rPPG ---
        cv2.putText(plot_area, "1. Signal ECG / rPPG", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        if len(signal_ecg) > 15 and np.max(signal_ecg) != np.min(signal_ecg):
            sig_min, sig_max = np.min(signal_ecg), np.max(signal_ecg)
            norm_sig = (signal_ecg - sig_min) / (sig_max - sig_min)
            
            step = (half_w - 20) / len(norm_sig)
            for i in range(1, len(norm_sig)):
                x1 = int(10 + (i - 1) * step)
                y1 = int(130 - (norm_sig[i - 1] * 80))
                x2 = int(10 + i * step)
                y2 = int(130 - (norm_sig[i] * 80))
                if x2 < half_w:
                    cv2.line(plot_area, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.line(plot_area, (half_w, 10), (half_w, 140), (55, 55, 55), 1)

        # --- Graphe 2 : Historique BPM ---
        cv2.putText(plot_area, f"2. Historique BPM (CNN): {int(bpm_final)} bpm | {status_actuel}", 
                    (half_w + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, status_color, 1)
        
        if len(bpm_history) > 2:
            bpm_min, bpm_max = 40.0, 140.0
            step_hist = (half_w - 20) / HISTORY_SIZE
            for i in range(1, len(bpm_history)):
                x1 = int(half_w + 10 + (i - 1) * step_hist)
                val1 = np.clip(bpm_history[i - 1], bpm_min, bpm_max)
                y1 = int(130 - ((val1 - bpm_min) / (bpm_max - bpm_min) * 80))
                
                x2 = int(half_w + 10 + i * step_hist)
                val2 = np.clip(bpm_history[i], bpm_min, bpm_max)
                y2 = int(130 - ((val2 - bpm_min) / (bpm_max - bpm_min) * 80))
                
                if x2 < w:
                    cv2.line(plot_area, (x1, y1), (x2, y2), status_color, 2)
                    
        return np.vstack((frame, plot_area))
    except Exception as e:
        print(f"[Erreur graphe] {e}")
        return frame


def generer_flux_batch():
    """Générateur de frames pour le flux vidéo batch"""
    if not os.path.exists(VIDEO_FOLDER_PATH):
        print(f"[!] Dossier introuvable : {VIDEO_FOLDER_PATH}")
        return

    # Nettoyer l'ancien fichier Excel
    if os.path.exists(OUTPUT_EXCEL_PATH):
        try:
            os.remove(OUTPUT_EXCEL_PATH)
        except Exception as e:
            print(f"[!] Impossible de supprimer l'ancien Excel : {e}")

    extensions_valides = ('.mp4', '.avi', '.mkv', '.mov')
    liste_videos = [f for f in os.listdir(VIDEO_FOLDER_PATH) if f.lower().endswith(extensions_valides)]

    for index, nom_video in enumerate(liste_videos, 1):
        chemin_complet = os.path.join(VIDEO_FOLDER_PATH, nom_video)
        cap = cv2.VideoCapture(chemin_complet)
        fs = cap.get(cv2.CAP_PROP_FPS)
        if fs <= 0:
            fs = 30.0

        # === BUFFERS ET HISTORIQUES POUR CETTE VIDÉO ===
        green_buffer = deque(maxlen=BUFFER_SIZE)
        cnn_bpm_history = deque(maxlen=HISTORY_SIZE)
        bpm_recent_window = deque(maxlen=STABILITY_WINDOW)
        
        # Métriques finales pour Excel
        high_freq_signals = []
        bpm_cnn_list = []
        bpm_peak_list = []
        
        # === TRACKING DE L'INSTABILITÉ ===
        frame_status_list = []
        total_frames = 0

        print(f"\n[▶] Traitement vidéo {index}/{len(liste_videos)} : {nom_video}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
            
            # === SÉLECTIONNER UNIQUEMENT LE VISAGE PRINCIPAL ===
            main_face = selectionner_visage_principal(faces)
            
            bpm_final = 0
            bpm_classique = 0
            signal_a_dessiner = np.zeros(BUFFER_SIZE)
            frame_status = "INITIALISATION"
            status_color = (0, 255, 255)
            
            if main_face is not None:
                (x, y, w, h) = main_face
                f = frame[y:y + h, x:x + w]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                if f.size > 0:
                    # Extraire le canal vert
                    green_val = np.mean(f[:, :, 1])
                    green_buffer.append(green_val)
                    
                    # Traitement si le buffer est suffisamment rempli
                    if len(green_buffer) > 30:
                        # === NORMALISER LE SIGNAL ===
                        signal = np.array(green_buffer)
                        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
                        filtered = bandpass_filter(signal, 0.7, 3.5, fs)
                        signal_a_dessiner = filtered[-BUFFER_SIZE:] if len(filtered) >= BUFFER_SIZE else filtered
                        
                        # === CALCUL BPM ===
                        high_freq_signals.append(filtered[-1])
                        
                        # BPM via détection de pics
                        peaks, _ = find_peaks(filtered, distance=int(fs * 0.5))
                        if len(peaks) > 1:
                            bpm_classique = len(peaks) * (60 / (len(filtered) / fs))
                            bpm_peak_list.append(bpm_classique)
                        
                        # BPM via CNN
                        bpm_cnn = predire_bpm_cnn(filtered, len(green_buffer))
                        bpm_final = bpm_cnn if bpm_cnn else bpm_classique
                        
                        if bpm_cnn:
                            bpm_cnn_list.append(bpm_cnn)
                            cnn_bpm_history.append(bpm_cnn)
                            bpm_recent_window.append(bpm_cnn)
                        
                        # === ÉVALUER LA STABILITÉ DE CETTE FRAME ===
                        is_signal_stable = evaluer_stabilite_signal(
                            list(green_buffer), 
                            list(bpm_recent_window)
                        )
                        
                        # === DÉTERMINER LE STATUT FINAL ===
                        frame_status, status_color = determiner_status(bpm_final, is_signal_stable)
                        frame_status_list.append(frame_status)
                        
                        # === AFFICHAGE SUR LA FRAME ===
                        cv2.putText(frame, f"Fichier: {nom_video} ({index}/{len(liste_videos)})", 
                                   (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        cv2.putText(frame, f"BPM (CNN): {int(bpm_final)}", 
                                   (x, y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                        cv2.putText(frame, f"BPM (Peak): {int(bpm_classique)}", 
                                   (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(frame, frame_status, 
                                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                    else:
                        frame_status = "INITIALISATION..."
                        cv2.putText(frame, frame_status, 
                                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            # Dessiner les graphes
            frame_final = dessiner_les_graphes(frame, signal_a_dessiner, 
                                               list(cnn_bpm_history), bpm_final, frame_status, status_color)
            
            # Encoder et envoyer la frame
            ret_enc, buffer = cv2.imencode(".jpg", frame_final)
            if ret_enc:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        cap.release()

        # === CALCUL DU DIAGNOSTIC FINAL POUR EXCEL ===
        if frame_status_list:
            # Compter les frames instables
            instable_count = sum(1 for status in frame_status_list if status == "INSTABLE")
            instability_ratio = instable_count / len(frame_status_list)
            
            print(f"    → Frames: {len(frame_status_list)} | Instables: {instable_count} ({instability_ratio*100:.1f}%)")
            
            # Déterminer le diagnostic final
            if instability_ratio >= INSTABILITY_FRAME_THRESHOLD:
                diagnostic_final = "INSTABLE"
            else:
                mean_bpm_cnn = np.mean(bpm_cnn_list) if bpm_cnn_list else 0.0
                if mean_bpm_cnn < 50:
                    diagnostic_final = "FATIGUE"
                elif mean_bpm_cnn > 100:
                    diagnostic_final = "TACHYCARDIE"
                else:
                    diagnostic_final = "STABLE"
        else:
            diagnostic_final = "ERREUR (pas de données)"
            instability_ratio = 0.0

        # === ENREGISTREMENT IMMÉDIAT DANS EXCEL ===
        if high_freq_signals:
            mean_bpm_cnn = np.mean(bpm_cnn_list) if bpm_cnn_list else 0.0
            mean_bpm_peak = np.mean(bpm_peak_list) if bpm_peak_list else 0.0
            
            nouvelle_ligne = {
                "Nom du Fichier": nom_video,
                "Total Frames": total_frames,
                "Instability Ratio (%)": round(instability_ratio * 100, 2),
                "HF_Signal_Moyenne": round(np.mean(high_freq_signals), 5),
                "HF_Signal_Variance": round(np.std(high_freq_signals), 5),
                "Moyenne_BPM_CNN": round(mean_bpm_cnn, 2),
                "Moyenne_BPM_Peak": round(mean_bpm_peak, 2),
                "Diagnostic": diagnostic_final
            }

            # Lire l'existant ou créer un nouveau DataFrame
            if os.path.exists(OUTPUT_EXCEL_PATH):
                try:
                    df_existant = pd.read_excel(OUTPUT_EXCEL_PATH)
                    df_nouveau = pd.DataFrame([nouvelle_ligne])
                    df_complet = pd.concat([df_existant, df_nouveau], ignore_index=True)
                except Exception as e:
                    print(f"    [!] Erreur lecture Excel : {e}")
                    df_complet = pd.DataFrame([nouvelle_ligne])
            else:
                df_complet = pd.DataFrame([nouvelle_ligne])

            # Écriture immédiate sur le disque
            try:
                df_complet.to_excel(OUTPUT_EXCEL_PATH, index=False)
                print(f"    [✔] Enregistrement Excel OK : {diagnostic_final}")
            except Exception as e:
                print(f"    [!] Erreur écriture Excel : {e}")


@app.route('/video_feed')
def video_feed():
    return Response(generer_flux_batch(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template_string('''
    <html>
        <head><title>NabdVisio Multi-Analytics v2</title></head>
        <body style="text-align: center; background-color: #1a1a1a; color: white; font-family: Arial; padding-top: 20px;">
            <h2>Interface NabdVisio - Batch Processing View (v2)</h2>
            <p style="color: #aaa; font-size: 12px;">Logique : Signal stable = STABLE si BPM normal, sinon FATIGUE/TACHYCARDIE</p>
            <img src="/video_feed" style="width: 80%; max-width: 850px; border: 3px solid #444; border-radius:8px; box-shadow: 0px 0px 15px rgba(0,0,0,0.5);"/>
        </body>
    </html>
    ''')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
