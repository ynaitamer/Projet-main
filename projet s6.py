import tkinter as tk

def calculate_result():
    try:
        input_value = float(entry.get())
        result = input_value * 2  # Exemple : Doubler la valeur entrée
        result_label.config(text=f"Résultat : {result}")
    except ValueError:
        result_label.config(text="Erreur : Entrée non valide")

# Crée la fenêtre principale
window = tk.Tk()
window.title("Interface Interactive")

# Crée une entrée Nombre de fonctionnaires
label1 = tk.Label(window, text="Nombre de fonctionnaires :")
label1.pack()
entry1 = tk.Entry(window)
entry1.pack()

# Crée une entrée Valeur du point d'indice
label2 = tk.Label(window, text="Valeur du point d'indice :")
label2.pack()
entry2 = tk.Entry(window)
entry2.pack()

# Crée un bouton pour lancer le calcul
calculate_button = tk.Button(window, text="Calculer", command=calculate_result)
calculate_button.pack()

# Crée une étiquette (Label) pour afficher les résultats
result_label = tk.Label(window, text="")
result_label.pack()

# Lance la boucle principale de l'interface utilisateur
window.mainloop()
