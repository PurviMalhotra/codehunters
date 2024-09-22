import subprocess
import time
import tkinter 
from tkinter import messagebox

# Run the first script
subprocess.run(["python3", "speed_identify_animals.py"])  # Replace 'python3' with 'python' for Windows if necessary

# Wait for 2 seconds before running the second script
time.sleep(2)

from speed_identify_animals2 import identified_labels

# Create the main window for the message box
root = tkinter.Tk()
root.withdraw()  # Hide the main window

# Show the pop-up message box
if 'bear' in identified_labels:
    messagebox.showinfo('Warning', 'OMG! Bear detected, RUNNNNNNN!!!')
else:
    messagebox.showinfo('Info', 'Animal detection completed!')

# Keep the program running until the message box is closed
root.mainloop()





