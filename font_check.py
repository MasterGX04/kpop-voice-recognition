import tkinter as tk
import tkinter.font as tkFont

root = tk.Tk()

uiFont = tkFont.Font(
    root=root,
    family="Pretendard Variable",
    size=18,
    weight="normal"
)

print(uiFont.actual())

tk.Label(root, text="Pretendard test 씨발 엄청 자쯩났어", font=uiFont).pack()
root.mainloop()