#!/bin/bash

# ------------------------
# Posture Reminder Script
# Sends a zenity popup every 5 minutes to remind user to sit straight.
# Auto-closes after timeout if no interaction.
# ------------------------
# TODO:
# - Add snooze functionality
# - Add terminate button to stop reminders

# Settings
INTERVAL=300  # 5 minutes in seconds
TIMEOUT=10    # Close the popup after 10 seconds

# Infinite loop
while true; do

    # Show popup with formatting (font size 14, red, italic)
    zenity --info \
        --title="🧘‍♂️ Posture Check" \
        --text="<span font='14' foreground='red'><i>Sit up straight and relax your shoulders!</i></span>" \
        --width=500 \
        --timeout=$TIMEOUT
        

    # Wait for next reminder
    sleep $INTERVAL
done