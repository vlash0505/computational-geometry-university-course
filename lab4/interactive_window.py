import tkinter as tk
from voronoi_fortune import Voronoi


class MainWindow:
    """
    This is the main window of the application which displays the Voronoi diagram
    as a result of user input points. User can also clear the screen and recalculate the Voronoi diagram.
    """
    RADIUS = 3  # Radius of drawn points on canvas

    def __init__(self, master):
        """
        Initializes the MainWindow object.

        Parameters:
        master (tk.Tk): The root window.
        """
        self.master = master
        self.master.title("Voronoi")

        # Create main frame
        self.frmMain = tk.Frame(self.master, relief=tk.RAISED, borderwidth=1)
        self.frmMain.pack(fill=tk.BOTH, expand=1)

        # Create canvas on the main frame
        self.w = tk.Canvas(self.frmMain, width=500, height=500)
        self.w.config(background='white')
        self.w.bind('<Double-1>', self.onDoubleClick)  # Bind double click event to onDoubleClick function
        self.w.pack()

        # Create frame for buttons
        self.frmButton = tk.Frame(self.master)
        self.frmButton.pack()

        # Create Calculate button and Clear button
        self.btnCalculate = tk.Button(self.frmButton, text='Calculate', width=25, command=self.onClickCalculate)
        self.btnCalculate.pack(side=tk.LEFT)

        self.LOCK_FLAG = False  # Initialize flag to lock the canvas when drawn

    def onClickCalculate(self):
        """
        Callback function for Calculate button click.
        """
        self.w.delete("lines")  # Delete lines from previous calculation
        if not self.LOCK_FLAG:
            self.LOCK_FLAG = False  # Reset the LOCK_FLAG

            # Find all points on canvas
            pObj = self.w.find_all()
            points = []
            for p in pObj:
                coord = self.w.coords(p)  # Get coordinates of point
                points.append((coord[0] + self.RADIUS, coord[1] + self.RADIUS))  # Adjust points according to radius

            vp = Voronoi(points)  # Initialize Voronoi object
            vp.process()  # Process the points
            lines = vp.get_output()  # Get output lines

            self.drawLinesOnCanvas(lines)  # Draw Voronoi lines on canvas

            print(lines)  # Print lines to console

    def onClickClear(self):
        """
        Callback function for Clear button click.
        """
        self.LOCK_FLAG = False  # Reset the LOCK_FLAG
        self.w.delete(tk.ALL)  # Clear the canvas

    def onDoubleClick(self, event):
        """
        Callback function for mouse double click event.

        Parameters:
        event (Event): The mouse event object.
        """
        if not self.LOCK_FLAG:
            self.w.delete("lines")  # Delete lines from previous calculation
            # Draw a point on canvas where user double clicked
            self.w.create_oval(event.x - self.RADIUS, event.y - self.RADIUS, event.x + self.RADIUS,
                               event.y + self.RADIUS, fill="black")

    def drawLinesOnCanvas(self, lines):
        """
        Draws lines on the canvas.

        Parameters:
        lines (list): List of line endpoints.
        """
        for l in lines:
            self.w.create_line(l[0], l[1], l[2], l[3], fill='blue', tags='lines')


def init():
    """
    Initialize the main window of the application.
    """
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()


if __name__ == '__main__':
    init()
