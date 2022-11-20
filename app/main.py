import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pages.loadmodelpage
import pages.trainorloadpage
import pages.trainmodelpage
import pages.welcomepage
import pages.singleormasspredictionpage
import pages.singlepredictionpage
import queue

from PySide2.QtCore import (
    QSize,
    Signal,
    QThread,
)
from PySide2.QtWidgets import (
    QApplication,
    QMainWindow
)

class WriteStream(object):
    def __init__(self,queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(text)

    def flush(self):
        pass

class StreamListener(QThread):
    text_signal = Signal(str)

    def __init__(self,queue,*args,**kwargs):
        QThread.__init__(self,*args,**kwargs)
        self.threadactive = True
        self.queue = queue

    def run(self):
        while self.threadactive:
            text = self.queue.get()
            self.text_signal.emit(text)

    def stop(self):
        self.threadactive = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emotify")
        self.setFixedSize(QSize(950, 700))

        self.workdir = "D:/emo/appfiles"

        self.welcome_page = pages.welcomepage.WelcomePage(self)
        self.train_or_load_page = pages.trainorloadpage.TrainOrLoadPage(self)
        self.train_model_page = pages.trainmodelpage.TrainModelPage(self)
        self.load_model_page = pages.loadmodelpage.LoadModelPage(self)
        self.single_or_mass_prediction_page = pages.singleormasspredictionpage.SingleOrMassPredictionPage(self)
        self.single_prediction_page = pages.singlepredictionpage.SinglePredictionPage(self)

        self.message_queue = queue.Queue()
        sys.stdout = WriteStream(self.message_queue)
        self.stream_listener = StreamListener(self.message_queue)
        self.stream_listener.text_signal.connect(self.train_model_page.append_text)
        self.stream_listener.start()

        self.show_page(self.welcome_page)


    def closeEvent(self, event):
        # safely stop training thread
        self.train_model_page.training_thread.quit()
        self.train_model_page.training_thread.wait()

        # change stdout back to normal
        sys.stdout = sys.__stdout__

        # put exit message in queue
        self.message_queue.put("exit")

        # safely stop stream listener
        self.stream_listener.stop()
        self.stream_listener.quit()
        self.stream_listener.wait()

        event.accept()
    
    def show_page(self, page):
        if self.centralWidget() is not None:
            self.centralWidget().setParent(None)
        self.setCentralWidget(page)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())