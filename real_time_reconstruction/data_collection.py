import numpy as np
from io_methods import read_interrogator_csv

### For now just methods to ensure data collected is valid
### Later on will be more of a UI based (similar to Jacynthe's) for easy data collection

def confirm_valid_reading(filename):
    try:
        read_interrogator_csv(filename, 14, 7)
        return True
    except ValueError as e:
        return False
    
def main():
    running = True
    fpath_confirm = False

    while not fpath_confirm:
        fpath = input("Enter filepath: ")
        confirmed = input(f"Is this correct: {fpath}? To make changes, enter N. Otherwise, hit enter: ")

        if confirmed != "N":
            fpath_confirm = True


    while running:
        fname_confirm = False
        while not fname_confirm:
            curve = input("Enter curvature: ")
            angle = input("Enter angle (degrees): ")
            insertion = input("Enter insertion #: ")

            filename = f'{fpath}Curvature_{curve}_Orientation_{angle}_{insertion}'
            #filename = f'{fpath}{curve}-{angle}-{insertion}'

            confirmed = input(f"Is this correct: {filename}? To make changes, enter N. Otherwise, hit enter: ")

            if confirmed != "N":
                fname_confirm = True
        
        result = confirm_valid_reading(filename)

        if result:
            print("Insertion was valid (200 valid readings obtained)")
        else:
            print("Insertion was INVALID (200 valid readings NOT present)")
        
        cont = input("Enter Q to exit. Hit enter to continue.")

        if cont == "Q":
            running = False

if __name__ == '__main__':
    main()
        




