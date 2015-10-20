
from Tkinter import *
import sys

the_word = sys.argv[1]
input_text = sys.argv[2]

class GrabberGui(Frame):
	def createWidgets(self):
		self.WORD = Message(self)
		self.WORD["text"] = the_word
		self.WORD["width"] = 300

		self.WORD.pack()

		self.TEXT = Message(self)
		self.TEXT["text"] = input_text
		self.TEXT["width"] = 300
		
		self.TEXT.pack({"side":"top"})

		self.INPUT = Entry(self)
		self.INPUT.insert(0, "Put source language here")

		self.INPUT.pack()

		self.SUBMIT = Button(self)
		self.SUBMIT["text"] = "Submit Custom Language"
		self.SUBMIT["command"] = self.send_user_text

		self.SUBMIT.pack()

		self.OE = Button(self)
		self.OE["text"] = "Old English"
		self.OE["command"] = self.OldEng

		self.OE.pack()

		self.OF = Button(self)
		self.OF["text"] = "Old French"
		self.OF["command"] = self.OldFr

		self.OF.pack()

		self.ME = Button(self)
		self.ME["text"] = "Middle English"
		self.ME["command"] = self.MidEng

		self.ME.pack()

		self.MF = Button(self)
		self.MF["text"] = "Middle French"
		self.MF["command"] = self.MidFr

		self.MF.pack()

		self.LAT = Button(self)
		self.LAT["text"] = "Latin"
		self.LAT["command"] = self.Latin

		self.LAT.pack()

		self.GRK = Button(self)
		self.GRK["text"] = "Greek"
		self.GRK["command"] = self.Greek

		self.GRK.pack()

		self.NOPE = Button(self)
		self.NOPE["text"] = "Skip"
		self.NOPE["command"] = self.Skip

		self.NOPE.pack({"side":"right"})

		self.QUIT = Button(self)
		self.QUIT["text"] = "Exit"
		self.QUIT["fg"] = "red"
		self.QUIT["command"] = self.quit 

		self.QUIT.pack({"side":"left"})

	def Skip(self):
		print "skip"
		self.quit()

	def OldEng(self):
		print "Old English"
		self.quit()

	def OldFr(self):
		print "Old French"
		self.quit()

	def MidEng(self):
		print "Middle English"
		self.quit()

	def MidFr(self):
		print "Middle French"
		self.quit()

	def Latin(self):
		print "Latin"
		self.quit()

	def Greek(self):
		print "Greek"
		self.quit()

	def send_user_text(self):
		temp = self.INPUT.get()
		print temp
		self.quit()




	def __init__(self, master=None):
		Frame.__init__(self, master)
		self.pack()
		self.createWidgets()


def main():
	the_word = sys.argv[1]
	input_text = sys.argv[2]
	root = Tk();
	app = GrabberGui( master = root)
	app.master.title("User Disambiguation Required")
	#app.master.geometry('{}x{}'.format(400,500))
	app.mainloop()
	root.destroy()

if __name__ == "__main__":
	main()