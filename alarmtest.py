import winsound  # For playing sound (Windows only)
import time
name='main'
# Function to determine severity level based on species and region
def classify_attack(animal, region_prone):
    # Define severity levels based on species
    high_severity_species = ["tiger", "bear", "leopard"]
    medium_severity_species = ["wolf", "crocodile"]
    low_severity_species = ["dog", "fox", "monkey"]
    
    # Check the severity based on the animal species
    if animal.lower() in high_severity_species:
        severity = 3
    elif animal.lower() in medium_severity_species:
        severity = 2
    elif animal.lower() in low_severity_species:
        severity = 1
    else:
        severity = 1  # Default to low severity if species is unknown

    # Increase severity if the region is prone to similar attacks
    if region_prone:
        severity += 1
        if severity > 3:
            severity = 3  # Cap severity to 3 (high)

    return severity

# Function to trigger an alarm based on severity level
def trigger_alarm(severity):
    if severity == 3:
        print("HIGH SEVERITY: Immediate danger! Evacuate!")
        # Play a sound alarm (you can replace this with a proper alarm system)
        for i in range(10):
            winsound.Beep(10000, 1000)  # Frequency 1000 Hz, duration 1 sec

    elif severity == 2:
        print("MEDIUM SEVERITY: Stay alert, be cautious!")
        for i in range(7):
            winsound.Beep(800, 5000)  # Frequency 800 Hz, duration 0.5 sec
    else:
        print("LOW SEVERITY: Stay aware, but no immediate danger.")
        for i in range(3):
            winsound.Beep(1000, 1000)  # Frequency 1000 Hz, duration 1 sec

# Example usage
def main():
    # Input from user (animal and region prone flag)
    animal = input("Enter the species of the animal: ")
    region_prone = input("Is the region prone to similar attacks? (yes/no): ").strip().lower() == "yes"

    # Classify severity
    severity = classify_attack(animal, region_prone)

    # Trigger an alarm based on severity
    trigger_alarm(severity)

if name == "main":
    main()
