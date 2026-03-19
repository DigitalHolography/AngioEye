from pathlib import Path

def collect_holo_paths():
    
    entree = input("Collez le chemin du dossier à scanner : ").strip()
    
    dossier_source = Path(entree.replace('"', ''))

    
    if not dossier_source.is_dir():
        print(f" Erreur : Le chemin '{dossier_source}' n'est pas un dossier valide.")
        return

    print(f" Analyse en cours de : {dossier_source}")
    
    
    fichiers_trouves = list(dossier_source.rglob('*.holo'))

    
    nom_resultat = "liste_chemins_holo.txt"
    
    try:
        with open(nom_resultat, 'w', encoding='utf-8') as f:
            for fichier in fichiers_trouves:
                
                f.write(f"{fichier.absolute()}\n")
        
        
        print("-" * 40)
        print(f" Opération réussie !")
        print(f" {len(fichiers_trouves)} fichiers trouvés et listés.")
        print(f" Fichier créé : {Path(nom_resultat).absolute()}")
        print("-" * 40)

    except Exception as e:
        print(f" Une erreur est survenue lors de l'écriture : {e}")

if __name__ == "__main__":
    collect_holo_paths()
