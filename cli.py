import sys
from stego import SafeNeuralStego, AVAILABLE_MODELS

def main():
    print("========================================")
    print("      NEUROSTEGO - CLI INTERFACE        ")
    print("========================================")
    
    print("\nAvailable Models:")
    models = list(AVAILABLE_MODELS.keys())
    for i, m in enumerate(models):
        print(f"{i+1}. {m}")
        
    choice = input("\nSelect Model (default 1): ")
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            model_name = models[idx]
        else:
            model_name = models[0]
    except:
        model_name = models[0]
        
    model_id = AVAILABLE_MODELS[model_name]
    print(f"\nInitializing {model_name}...")
    
    try:
        stego = SafeNeuralStego(model_id)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    while True:
        print("\n------------------------------")
        print("1. Encode (Hide Secret)")
        print("2. Decode (Reveal Secret)")
        print("3. Exit")
        
        mode = input("Select Option: ")
        
        if mode == '1':
            secret = input("\nEnter Secret Message: ")
            start = input("Start Text (Press Enter for default): ")
            if not secret: continue
            
            try:
                res = stego.encode(secret.encode('utf-8'), start)
                print("\n--- GENERATED COVER TEXT (COPY BELOW) ---")
                print(res)
                print("-----------------------------------------")
            except Exception as e:
                print(f"Error: {e}")
                
        elif mode == '2':
            print("\nEnter the full cover text (Press Enter twice to finish):")
            lines = []
            while True:
                line = input()
                if line == "": break
                lines.append(line)
            full_text = " ".join(lines)
            
            start = input("Start Text (Must match encoding): ")
            
            try:
                res = stego.decode(full_text, start)
                print(f"\n🔓 DECODED SECRET: {res.decode('utf-8')}")
            except Exception as e:
                print(f"Error: {e}")
                
        elif mode == '3':
            break

if __name__ == "__main__":
    main()
