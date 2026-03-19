import shutil
from pathlib import Path
from datetime import datetime

def collecteur_h5_final():
    print("============================================================")
    print("   COLLECTEUR H5 - NOM D'ORIGINE CONSERVÉ")
    print("============================================================\n")

    # 1. PARAMÈTRES
    txt_input = input("1. Glissez votre fichier .txt ici : ").strip()
    chemin_liste = Path(txt_input.replace('"', ''))
    
    if not chemin_liste.exists():
        print(f" Erreur : Le fichier '{chemin_liste}' est introuvable.")
        return

    today_str = datetime.now().strftime("%d/%m/%Y")
    date_input = input(f"2. Date de calcul [Défaut: {today_str}] : ").strip() or today_str
    try:
        date_cible = datetime.strptime(date_input, "%d/%m/%Y").date()
    except ValueError:
        print(" Format de date incorrect.")
        return

    dest_input = input("3. Dossier de destination : ").strip()
    dossier_dest = Path(dest_input.replace('"', ''))

    # 2. ANALYSE
    print("\n Analyse des dossiers...")
    
    with open(chemin_liste, 'r', encoding='utf-8') as f:
        chemins_sources = [ligne.strip() for ligne in f if ligne.strip()]

    fichiers_a_copier = []

    for chemin in chemins_sources:
        p = Path(chemin)
        nom_base = p.stem         
        dossier_parent = p.parent 

        # Ta structure : Nom_HD/eyeflow/Nom_EF/h5/nom_output.h5
        pattern = f"{nom_base}_HD_*/eyeflow/*_EF_*/h5/*.h5"
        
        for h5_file in dossier_parent.glob(pattern):
            # Filtre par date
            date_modif = datetime.fromtimestamp(h5_file.stat().st_mtime).date()

            if date_modif == date_cible:
                fichiers_a_copier.append(h5_file)

    # 3. VÉRIFICATION AVANT COPIE
    if not fichiers_a_copier:
        print(f"\n Aucun fichier trouvé pour la date du {date_cible}.")
        return

    print("\n" + "!" * 65)
    print(f"  VÉRIFICATION : {len(fichiers_a_copier)} FICHIERS À COPIER")
    print("!" * 65)
    
    for i, h5_file in enumerate(fichiers_a_copier, 1):
        taille_mo = round(h5_file.stat().st_size / (1024**2), 2)
        print(f" [{i}] NOM RÉEL : {h5_file.name}")
        print(f"     TAILLE   : {taille_mo} Mo")
        print(f"     SOURCE   : {h5_file.parent}")
        print("-" * 40)

    # 4. CONFIRMATION ET COPIE
    confirmation = input("\n Voulez-vous copier ces fichiers sans changer leurs noms ? (oui/non) : ").lower().strip()

    if confirmation == 'oui':
        dossier_dest.mkdir(parents=True, exist_ok=True)
        print("\n Copie en cours...")
        
        for h5_file in fichiers_a_copier:
            # ICI : On garde strictement le nom du fichier trouvé
            cible = dossier_dest / h5_file.name
            
            try:
                shutil.copy2(h5_file, cible)
                print(f" Copié : {h5_file.name}")
            except Exception as e:
                print(f" Erreur sur {h5_file.name} : {e}")
        
        print(f"\n Succès ! Les fichiers originaux sont dans : {dossier_dest.absolute()}")
    else:
        print("\n Annulé.")

if __name__ == "__main__":
    collecteur_h5_final()