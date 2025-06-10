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

    while running:
        curve = float(input("Enter curvature: "))
        angle = int(input("Enter angle (degrees): "))
        insertion = int(input("Enter insertion #: "))

        filename = f'{curve}-{angle}-{insertion}'

        result = confirm_valid_reading(filename)

        if result:
            print("Insertion was valid (200 valid readings obtained)")
        else:
            print("Insertion was INVALID (200 valid readings NOT present)")
        
        cont = input("Press Q to exit")

        if cont == "Q":
            running = False

if __name__ == '__main__':
    main()
        




