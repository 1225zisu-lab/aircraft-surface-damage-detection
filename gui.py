import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import os

model = YOLO("best.pt")

def select_image():
	global panel
	
	file_path = filedialog.askopenfilename()
	if not file_path:
		return

	img = cv2.imread(file_path)
	results = model(img)
	annotated = results[0].plot()

	save_path = "gui_output.jpg"
	cv2.imwrite(save_path, annotated)

	pil_img = Image.open(save_path)
	pil_img = pil_img.resize((600,450))
	img_tk = ImageTk.PhotoImage(pil_img)
	
	panel.configure(image=img_tk)
	panel.image = img_tk

root = tk.Tk()
root.title("Defect Detection GUI")

btn = tk.Button(root, text="Select Image", command=select_image)
btn.pack()

panel = tk.Label(root)
panel.pack()

root.mainloop()

