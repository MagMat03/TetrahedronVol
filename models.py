from sklearn.neural_network import MLPClassifier

# Inicjalizacja klasyfikatora MLP (Wielowarstwowy Perceptron)
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # Dwie warstwy ukryte: 128 i 64 neurony
    activation='relu',             # Funkcja aktywacji: Rectified Linear Unit (standard dla warstw ukrytych)
    solver='adam',                 # Optymalizator wag: Adam (szybki i wydajny dla większości zadań)
    alpha=0.0001,                  # Parametr regularyzacji L2 (zapobiega przeuczeniu/overfittingowi)
    batch_size='auto',             # Automatyczny dobór rozmiaru paczki (batcha)
    learning_rate='constant',      # Stały współczynnik uczenia
    learning_rate_init=0.001,      # Początkowy współczynnik uczenia
    max_iter=500,                  # Maksymalna liczba epok (przejść przez cały zbiór treningowy)
    early_stopping=True,           # Wcześniejsze zatrzymanie, jeśli model przestaje się uczyć na zbiorze walidacyjnym
    validation_fraction=0.1,       # 10% danych treningowych zostanie użyte jako zbiór walidacyjny do early stoppingu
    random_state=42,               # Ziarno losowości, aby wyniki eksperymentu były powtarzalne
    verbose=True                   # Wyświetla postęp uczenia (epoka po epoce) w konsoli
)