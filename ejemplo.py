"""This is an example module. Add your module description here."""

# Rest of your code follows
import json

import requests


class DataProcessor:
    """Clase para procesar datos."""


def __init__(self, data):
    self.data = data


def filter_data(self, keyword):
    """Filtrar datos por palabra clave."""
    filtered_data = [item for item in self.data if keyword in item["name"]]
    return filtered_data


def fetch_data_from_api(url: str) -> list:
    """Obtiene datos de la API y devuelve una lista."""
    response = requests.get(url)
    return response.json()


def save_data_to_file(file_path: str, data: dict) -> None:
    """Guarda los datos en un archivo JSON."""
    with open(file_path, "w") as file:
        json.dump(data, file)


def print_welcome_message():
    """Imprime mensaje de bienvenida."""
    # Imprime un mensaje de bienvenida
    print("Bienvenido al procesador de datos. Preparando para iniciar...")


# URL de ejemplo para la API
if __name__ == "__main__":
    """Inicializa la clase con los datos proporcionados."""
    api_url = "https://api.example.com/data"
    data = fetch_data_from_api(api_url)
    data_processor = DataProcessor(data)
    filtered_data = data_processor.filter_data("especial")
    save_data_to_file("/mnt/data/filtered_data.json", filtered_data)
    print_welcome_message()
