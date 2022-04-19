# -*- coding: utf-8 -*-
import sys  
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel, QLineEdit, QTextBrowser, QFileDialog, QMessageBox, QCheckBox, QComboBox
from PyQt5.QtCore import QBasicTimer
import time
import random
import requests
import hashlib
import json


class FirstUi(QMainWindow):  
    def __init__(self):
        super(FirstUi, self).__init__()
        self.flag = False
        self._count = 15
        self._echo = ''
        self.lang = 1
        self.balance = 20
        self.model = ''
        self.data = ''
        self.current_taskhash = ''
        self.waitres = False
        self.serverip = 'http://127.0.0.1:7000'
        self.r = hashlib.sha256(bytes(str(random.random()), encoding='utf8')).hexdigest()
        self.init_ui()
        self.itemcount = 0

    def init_ui(self):
        self.resize(800, 400)  
        self.setWindowTitle('Data Node')  

        self._lable4 = QLabel('<h1>PoLe Demo</h1>', self)
        self._lable4.setGeometry(330, 20, 300, 25)

        self._lable6 = QLabel('Online : {}'.format(self.serverip), self)
        self._lable6.setGeometry(550, 15, 300, 25)

        self.btn = QPushButton('Change Server', self) 
        self.btn.setGeometry(660, 45, 125, 30) 
        self.btn.clicked.connect(self.changeserverfunc)

        self.le = QLineEdit(self)
        self.le.setGeometry(550, 45, 100, 25)
        self.le.setText('http://')

        self.lable1 = QLabel('Select Dataset:', self)
        self.lable1.setGeometry(320, 260, 300, 25)

        # 数据集
        self.item = ['Select...', 'MNIST', 'Fashion-MNIST', 'CIFAR-100', 'CIFAR-10', 'IRIS', 'IMDB']

        self.combo = QComboBox(self)
        for i in range(0, len(self.item)):
            self.combo.addItem(str(self.item[i]))
        self.combo.activated[str].connect(self.onActivated)

        self.combo.move(420, 260)

        self.btn3 = QPushButton('Push Training Request', self)  
        self.btn3.setGeometry(310, 310, 200, 50)  
        self.btn3.clicked.connect(self.push) 

        self._lable1 = QLabel('To update data set, click here:', self)
        self._lable1.setGeometry(320, 290, 300, 25)

        self._lable2 = QLabel('Pay Incentives:', self)
        self._lable2.setGeometry(320, 190, 300, 25)

        self._lable3 = QLabel('Choose a ML model description file:', self)
        self._lable3.setGeometry(320, 110, 300, 25)


        self.item = ['Custom', "LeNet-5", 'VGG-16', 'VGG-19', 'resNet', 'Config...']

        self.combo2 = QComboBox(self)
        for i in range(0, len(self.item)):
            self.combo2.addItem(str(self.item[i]))
        self.combo2.activated[str].connect(self.onActivated2)

        self.combo2.setGeometry(440, 150, 105, 25)

        self.le2 = QLineEdit(self)
        self.le2.setGeometry(320, 220, 100, 25)
        self.le2.setText('')

        self.lable5 = QLabel('<h2>Balance:</h2>', self)
        self.lable5.setGeometry(30, 50, 300, 25)

        self.lable6 = QLabel('<h2>Address:</h2>', self)
        self.lable6.setGeometry(30, 200, 300, 25)

        self.lable2 = QLabel('<h1>{}</h1>'.format(self.balance), self)
        self.lable2.setGeometry(30, 90, 300, 25)

        self.lable3 = QLabel('', self)
        self.lable3.setGeometry(250, 150, 70, 25)

        self.lable4 = QLabel('', self)
        self.lable4.setGeometry(250, 265, 70, 25)

        self.timer = QBasicTimer()  
        self.timer.start(1000, self)

        self._lable = QLabel('<h1>{}</h1>'.format(self.r[:10]), self)
        self._lable.setGeometry(30, 240, 300, 25)

        self.tb = QTextBrowser(self)
        self._echo = """Received:
    Number      Time          Accuracy    
        """
        self.tb.setText(self._echo)
        self.tb.setGeometry(550, 100, 245, 220)

        self.cb = QCheckBox(self)
        self.cb.setText('Encrypt')
        self.cb.move(320, 350)


        self.btn5 = QPushButton('Refresh', self)  
        self.btn5.setGeometry(560, 330, 100, 25) 
        self.btn5.clicked.connect(self.slot_btn_function) 


    def slot_btn_function(self):

        pass

    def choose_dataset(self):
        openfile_name = QFileDialog.getOpenFileName(self, '选择文件', '', '*.*')
        if openfile_name:
            self.lable4.setText("(Selected)")

    def onActivated(self, text):
        self.lable4.setText("(Selected)")
        self.data = text

    def onActivated2(self, text):
        self.lable3.setText("(Selected)")
        self.model = text

    def push(self):
            # 支付的钱>余额
            if int(self.le2.text()) > self.balance:
                QMessageBox.information(self, "Error", "Insufficient fund.", QMessageBox.Ok)
                return
            else:
                self.balance -= int(self.le2.text())
            self.lable2.setText('<h1>{}</h1>'.format(self.balance))
            self.current_taskhash = hashlib.md5(bytes(str(random.random()), encoding='utf8')+bytes(self.model, encoding='utf8')+bytes(self.data, encoding='utf8')).hexdigest()
            request_dict = {'balance': int(self.le2.text()),
                            'acc': "%",
                            'data': self.data,  # 数据集
                            'model': self.model,
                            'taskhash': self.current_taskhash,
                            'addr': self.r}
            print('[REQUEST CONTENT]:{}'.format(request_dict))
            print('[TASK HASH]:{}'.format(self.current_taskhash))
            try:
                requests.post('{}/get-request'.format(self.serverip), json=request_dict)
                QMessageBox.information(self, "TIPS", "Push successful.", QMessageBox.Ok)
                self.flag = True
                self.lable4.setText("")
                self.lable3.setText("")
                self.waitres = True
            except Exception as e:
                print('[E1]:{}'.format(e))
                QMessageBox.warning(self, "ERROR", "Check connections.", QMessageBox.Ok)




    def timerEvent(self, event):
        if self.waitres:
            try:
                req = requests.get('{}/datanode/ifdone/{}'.format(self.serverip, self.current_taskhash))
                jsonres = json.loads(req.content)
                # 完成
                if jsonres['code'] == 200:
                    print('[BLOCKIDX]:{}'.format(jsonres['blockidx']))
                    QMessageBox.information(self, "INFO", "Received new model!", QMessageBox.Ok)
                    self.append_echo(jsonres)
                    self.waitres = False
                else:
                    return

            except Exception as e:
                print('[E2]:{}'.format(e))
        else:
            return


    def append_echo(self, jsonres):
        print(jsonres)
        self.itemcount += 1
        self._echo += """{} | {} | {}%
---------------------------------- 
        """.format(self.itemcount, time.time(), int(float(jsonres["acc"])*100))
        self.tb.setText(self._echo)

    def changeserverfunc(self):
        self.serverip = self.le.text()
        print('[Switch Server]:{}'.format(self.serverip))
        if self.lang == 0:
            self._lable6.setText('在线 : {}'.format(self.serverip))
        else:
            self._lable6.setText('Online : {}'.format(self.serverip))


def main():
    app = QApplication(sys.argv)
    w = FirstUi()  
    w.show() 
    sys.exit(app.exec_()) 


if __name__ == '__main__':  
    main()

