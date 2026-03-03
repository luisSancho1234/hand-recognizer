import tkinter as tk
from tkinter import ttk

class App(tk.Tk):
    def __init__(self, logic_module):
        super().__init__()

        self.logic = logic_module
        self.title("Hand Recognizer GUI")
        self.geometry("400x200")

        # Label de status
        self.label = ttk.Label(self, text="Aguardando reconhecimento...")
        self.label.pack(pady=10)

        # Botão para rodar reconhecimento
        self.btn = ttk.Button(self, text="Rodar Reconhecimento", command=self.run_recognition)
        self.btn.pack(pady=10)

    def run_recognition(self):
        result = self.logic.reconhecer_mao()
        self.label.config(text=f"Resultado: {result}")