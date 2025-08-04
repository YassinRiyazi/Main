
## Learned

### Making Window on top and fullscreen
> [!CAUTION]
> Add toggle to walk out of full screen
```Python
window_name = f'{_row}x{_col} Video Grid'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
# Set window to full screen
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# Set window to stay on top
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
font = cv2.FONT_HERSHEY_SIMPLEX
```

### Making full screen and vise versa

```Python
elif key == ord('f'):  # Toggle full screen with 'f' key
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
```