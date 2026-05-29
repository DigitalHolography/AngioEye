import matplotlib.pyplot as plt
import numpy as np


def main():
    r0 = [
        15.1,
        15.4,
        16.0,
        13.6,
        15.4,
        12.5,
        15.3,
        13.0,
        12.2,
        13.1,
        13.8,
        11.7,
        14.0,
        11.0,
        13.1,
        14.6,
        14.9,
        16.4,
        12.2,
        10.8,
        10.7,
        10.7,
        10.8,
        11.8,
        11.1,
        10.6,
        11.8,
        9.7,
        10.8,
        9.5,
        10.1,
        7.7,
        7.5,
        10.2,
        10.2,
        8.0,
        10.2,
        11.5,
        7.4,
        10.4,
        11.8,
        12.3,
        10.3,
    ]

    A = [
        -0.14,
        -0.12,
        -0.105,
        -0.14,
        -0.10,
        -0.13,
        -0.07,
        -0.14,
        -0.12,
        -0.10,
        -0.080,
        -0.100,
        -0.150,
        -0.220,
        -0.160,
        -0.160,
        -0.160,
        -0.120,
        -0.180,
        -0.200,
        -0.180,
        -0.160,
        -0.130,
        -0.150,
        -0.120,
        -0.120,
        -0.110,
        -0.160,
        -0.150,
        -0.150,
        -0.110,
        -0.170,
        -0.170,
        -0.120,
        -0.110,
        -0.130,
        -0.050,
        -0.060,
        -0.110,
        -0.100,
        -0.040,
        -0.040,
        -0.060,
    ]

    marque = [
        "030",
        "031",
        "032",
        "033",
        "034",
        "035",
        "036",
        "045",
        "046",
        "047",
        "048",
        "049",
        "052",
        "053",
        "054",
        "060",
        "061",
        "062",
        "070",
        "071",
        "072",
        "073",
        "074",
        "082",
        "083",
        "084",
        "085",
        "086",
        "087",
        "088",
        "089",
        "094",
        "095",
        "096",
        "097",
        "098",
        "099",
        "0104",
        "0105",
        "0106",
        "0107",
        "0108",
        "0109",
    ]

    x0 = [
        7.0,
        7.0,
        8.0,
        8.0,
        7.5,
        8.0,
        10.0,
        5.8,
        5.2,
        7.0,
        7.2,
        7.3,
        8.7,
        7.5,
        7.5,
        9.0,
        8.0,
        7.0,
        8.0,
        7.0,
        6.5,
        6.0,
        7.7,
        9.0,
        8.2,
        6.6,
        8.0,
        7.0,
        8.0,
        8.3,
        8.6,
        8.0,
        6.2,
        8.0,
        8.0,
        6.8,
        9.0,
        10.0,
        8.0,
        6.2,
        7.5,
        6.3,
        7.4,
    ]

    y0 = [
        8.0,
        7.1,
        6.7,
        6.45,
        5.90,
        5.05,
        4.10,
        5.90,
        4.60,
        4.30,
        3.80,
        3.45,
        7.40,
        6.70,
        6.85,
        8.50,
        8.90,
        8.10,
        6.70,
        5.80,
        5.15,
        4.55,
        3.80,
        5.20,
        3.70,
        3.35,
        3.82,
        3.80,
        4.35,
        3.38,
        2.80,
        2.52,
        2.42,
        3.10,
        2.85,
        2.08,
        1.30,
        2.00,
        1.50,
        2.70,
        1.40,
        1.52,
        1.60,
    ]

    def tri_fusion(liste):
        if len(liste) <= 1:
            return liste

        milieu = len(liste) // 2
        gauche = tri_fusion(liste[:milieu])
        droite = tri_fusion(liste[milieu:])

        return fusion(gauche, droite)

    def fusion(gauche, droite):
        resultat = []
        i = 0
        j = 0

        while i < len(gauche) and j < len(droite):
            if gauche[i]["r0"] <= droite[j]["r0"]:
                resultat.append(gauche[i])
                i += 1
            else:
                resultat.append(droite[j])
                j += 1

        while i < len(gauche):
            resultat.append(gauche[i])
            i += 1

        while j < len(droite):
            resultat.append(droite[j])
            j += 1

        return resultat

    def moyenne(liste):
        if len(liste) == 0:
            return None
        return sum(liste) / len(liste)

    def mediane(liste_triee):
        n = len(liste_triee)

        if n == 0:
            return None

        milieu = n // 2

        if n % 2 == 1:
            return liste_triee[milieu]
        else:
            return (liste_triee[milieu - 1] + liste_triee[milieu]) / 2

    def tracer_histogramme(liste, largeur_bloc=1):
        if len(liste) == 0:
            print("La liste est vide.")
            return

        minimum = min(liste)
        maximum = max(liste)

        bornes = np.arange(minimum, maximum + largeur_bloc, largeur_bloc)

        liste_triee = sorted(liste)
        moy = moyenne(liste)
        med = mediane(liste_triee)

        plt.hist(liste, bins=bornes, edgecolor="black", zorder=1)

        if moy is not None:
            plt.axvline(
                moy,
                color="black",
                linestyle="--",
                linewidth=2,
                label=f"Moyenne = {moy:.2f}",
                zorder=10,
            )

        if med is not None:
            plt.axvline(
                med,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Médiane = {med:.2f}",
                zorder=10,
            )

        plt.xlabel("Valeurs")
        plt.ylabel("Effectifs")
        plt.title("Histogramme avec moyenne et médiane")
        plt.legend()
        plt.show()

    def filtre_donnees(donnees, moyenne_r0, mediane_r0, n):
        donnees_filtrees = []

        if moyenne_r0 is None or mediane_r0 is None:
            return donnees_filtrees

        ecart = abs(moyenne_r0 - mediane_r0)

        borne_min = moyenne_r0 - n * ecart
        borne_max = moyenne_r0 + n * ecart

        for d in donnees:
            r = d["r0"]

            if borne_min < r < borne_max:
                donnees_filtrees.append(d)

        return donnees_filtrees

    def std(liste):
        moy = moyenne(liste)

        if moy is None:
            return None

        variance = sum((x - moy) ** 2 for x in liste) / len(liste)
        return variance**0.5

    def profile_analysis():
        donnees = []

        for m, a, x, y, r in zip(marque, A, x0, y0, r0, strict=True):
            donnees.append(
                {
                    "marque": m,
                    "A": a,
                    "x0": x,
                    "y0": y,
                    "r0": r,
                }
            )

        donnees = tri_fusion(donnees)

        r0_trie = [d["r0"] for d in donnees]

        moyenne_r0 = moyenne(r0_trie)
        mediane_r0 = mediane(r0_trie)

        donnees_filtrees = filtre_donnees(donnees, moyenne_r0, mediane_r0, 13)

        r0_filtre = [d["r0"] for d in donnees_filtrees]
        r0_std = std(r0_filtre)

        print("Moyenne r0 =", moyenne_r0)
        print("Médiane r0 =", mediane_r0)
        print("Nombre avant filtre =", len(donnees))
        print("Nombre après filtre =", len(donnees_filtrees))

        print("\nDonnées filtrées :")
        for d in donnees_filtrees:
            print(d)

        # tracer_histogramme(r0_trie, 1)
        # tracer_histogramme(r0_filtre, 0.5)

        values = np.array(
            [[d["A"], d["x0"], d["y0"], d["r0"]] for d in donnees_filtrees]
        )

        return values, r0_std

    # resultat = profile_analysis()
    print(moyenne(x0))
    print(mediane(x0))


if __name__ == "__main__":
    main()
