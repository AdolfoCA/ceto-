import matplotlib.pyplot as plt
import os

# Lista per memorizzare i tempi di computation
tempi_computation = []
key_size = None
encryption_enabled = False
csv_file = 'test79.csv'

# Apri il file e cerca i valori di computation time e informazioni sull'encryption
try:
    with open(csv_file, 'r') as file:
        for linea in file:
            # Cerca informazioni sull'encryption nelle righe di commento
            if linea.startswith('# USE_ENCRYPTION:'):
                encryption_value = linea.split(':')[1].strip()
                encryption_enabled = (encryption_value.lower() == 'true')
            elif linea.startswith('# KEY_SIZE:'):
                try:
                    key_size = int(linea.split(':')[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            
            # Salta le righe di commento e intestazione
            if linea.startswith('#') or linea.startswith('Time'):
                continue
            
            # Dividi la linea per virgole
            parti = linea.strip().split(',')
            
            # Prendi l'ultimo valore come tempo di computation
            if len(parti) >= 1:
                try:
                    # Il tempo di computation è l'ultimo valore
                    computation_time = float(parti[-1])
                    tempi_computation.append(computation_time)
                except (ValueError, IndexError):
                    pass
except FileNotFoundError:
    print(f"Errore: File 'test62.csv' non trovato. Directory corrente: {os.getcwd()}")
    exit(1)
except Exception as e:
    print(f"Errore durante la lettura del file: {e}")
    exit(1)

# Se abbiamo trovato tempi di computation, crea il grafico
if tempi_computation:
    # Crea gli indici per le iterazioni
    iterazioni = range(1, len(tempi_computation) + 1)
    
    try:
        # Crea il grafico
        plt.figure(figsize=(10, 6))
        plt.plot(iterazioni, tempi_computation, marker='o')
        
        # Aggiungi titoli e etichette
        plt.title(f'Computation Time for Each Iteration (Encrypted: {encryption_enabled})')
        plt.xlabel('Iteration number')
        plt.ylabel('Computational time (ms)')
        plt.grid(True)
        
        # Aggiungi nota sulla dimensione della chiave se l'encryption è abilitata
        if encryption_enabled and key_size:
            plt.figtext(0.5, 0.01, f'Encryption enabled with {key_size}-bit key size',
                       ha='center', fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        elif encryption_enabled:
            plt.figtext(0.5, 0.01, 'Encryption enabled',
                       ha='center', fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        # Aggiungi media
        media = sum(tempi_computation) / len(tempi_computation)
        plt.axhline(y=media, color='red', linestyle='--', label=f'Average: {media:.4f} ms, Frequency:{100/media:.4f} Hz')
        plt.legend()
        
        # Regola i margini per fare spazio alla nota
        plt.subplots_adjust(bottom=0.15)
        
        # Crea la cartella enc_plots se non esiste
        plots_dir = 'enc_plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
            print(f"Cartella '{plots_dir}' creata.")
        
        # Genera il nome del file
        filename = f"{'encrypted' if encryption_enabled else 'unencrypted'}"
        if key_size and encryption_enabled:
            filename += f"_{key_size}bit"
        filename += "_computation_time.png"
        
        # Salva il grafico nella cartella enc_plots
        save_path = os.path.join(plots_dir, filename)
        plt.savefig(save_path)
        print(f"Grafico salvato con successo come: {save_path}")
        
        # Mostra il grafico
        plt.show()
        
        # Calcola statistiche
        print(f"Tempo medio di computation: {media:.4f} ms")
        print(f"Tempo minimo: {min(tempi_computation):.4f} ms")
        print(f"Tempo massimo: {max(tempi_computation):.4f} ms")
        
        if encryption_enabled:
            print(f"Encryption abilitata" + (f" con chiave di {key_size} bit" if key_size else ""))
            
    except Exception as e:
        print(f"Errore durante la creazione o il salvataggio del grafico: {e}")
else:
    print("Non sono stati trovati tempi di computation nel file.")