# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import keras

import sys  
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel, QLineEdit, QTextBrowser
from PyQt5.QtCore import *
import time
import random
import hashlib
import threading
import requests
from flask import *


def leNet5(input_shape):
    model = Sequential([
        Conv2D(filters=6,kernel_size=(5,5),padding="valid",activation="tanh",input_shape=input_shape),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(filters=16,kernel_size=(5,5),padding="valid",activation="tanh"),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(120,activation="tanh"),
        Dense(84,activation="tanh"),
        Dense(10,activation="softmax")
    ])
    
    return model


def vgg16(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model


class Worker(QThread):
    sinOut = pyqtSignal(str) 

    def __init__(self, parent=None):
        global app
        super(Worker, self).__init__(parent)
        self.app = app
        self.working = True
        self.port = -1
        self.addr = ''
        self.ipaddr = ''
        self.num = 0
        self.block = []

        # 准确率
        self.acc = 0

        # 节点列表
        self.node_list = []
        self.taskhashlist = []


        self.block_cache = []
        # 训练请求队列
        self.requestli_cache = []

    def __del__(self):
        self.working = False
        self.wait()

    def register(self):
        print('NODE LIST {}'.format(self.node_list))
        for ip in self.node_list:
            if ip == self.ipaddr:
                continue
            try:
                requests.post('{}/pole/protocol/addnode'.format(ip), json={'ipaddr':self.ipaddr, 'selfaddr': self.addr})
            except Exception as e:
                print('[E1]:{}'.format(e))

    def configargs(self, port, nodes, addr, blocks):
        self.port = port
        self.node_list = nodes
        self.addr = addr
        self.ipaddr = 'http://127.0.0.1:{}'.format(self.port)
        self.node_list.append(self.ipaddr)
        self.block = blocks
        self.register()


    def train(self):
        if len(self.requestli_cache)==0:
            return
        # 训练
        # 选择数据集
        if self.requestli_cache[0]["data"]=="CIFAR-10":
            print("载入数据集CIFAR-10")
            from keras.datasets import cifar10
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            inputshape = (32, 32, 3)
        elif self.requestli_cache[0]["data"]=="CIFAR-100":
            print("载入数据集CIFAR-100")
            from keras.datasets import cifar100
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()
            inputshape = (32, 32, 3)
        elif self.requestli_cache[0]["data"]=="CIFAR-100":
            print("载入数据集MNIST")
            from keras.datasets import mnist
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            inputshape = (28, 28)
        else:
            print("test")
            print(self.requestli_cache[0]["data"])
        
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        x_train = x_train / 255.0
        x_test = x_test / 255.0

        # one-hot独热映射
        y_label_train = keras.utils.to_categorical(y_train, 10)
        y_label_test = keras.utils.to_categorical(y_test, 10)

        # 选择模型
        if self.requestli_cache[0]["model"]=="LeNet-5":
            print("加载模型leNet5")
            model = leNet5(inputshape)
        elif self.requestli_cache[0]["model"]=="VGG-16":
            print("加载模型vgg16")
            model = vgg16(inputshape)

        # 模型编译
        model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy',
                        metrics=['accuracy'])

        # 模型训练
        model.fit(x_train, y_label_train, epochs=10)
        _, self.acc = model.evaluate(x_test, y_label_test)


    def generate_block(self):
        if not self.requestli_cache:
            self.requestli_cache = [{'balance': 0,
                                     'data':'Default',
                                     'model': None,
                                     'addr': None,
                                     'taskhash': None}]
        

        self_block = {'acc': self.acc,
                      'timestamp':time.time(),
                      'addr': self.addr,
                      'donetask': self.requestli_cache[0],
                      'body':{'msg': self.requestli_cache}}
        self.block_cache.append(self_block)
        self.broadcast(self_block)
        pass

    # 广播给其他共识节点
    def broadcast(self, selfblock):
        for ip in self.node_list:
            if ip == self.ipaddr:
                continue
            try:
                requests.post('{}/pole/protocol/consensus'.format(ip), json=selfblock)
            except Exception as e:
                print('[E2]:{}'.format(e))
        return

    def get_winner(self):
        print('BLOCK CACHE: {}'.format(self.block_cache))
        if not self.block_cache:
            return
        maxacc = -1
        winner_addr = ''
        bidx = None
        for idx in range(len(self.block_cache)):
            print('SEE', idx, self.block_cache[idx]['acc'], maxacc)
            if self.block_cache[idx]['acc'] > maxacc:
                maxacc = self.block_cache[idx]['acc']
                winner_addr = self.block_cache[idx]['addr']
                bidx = idx

        if winner_addr == self.addr:
            print('[Winner]:{}'.format(self.block_cache[bidx]))
            self.sinOut.emit('TOKENS:{}'.format(self.block_cache[bidx]['body']['msg'][0]['balance']))
        if len(self.requestli_cache) >= 1:
            # 第一个请求的任务完成，删除该任务
            self.requestli_cache = self.block_cache[bidx]['body']['msg'][1:]
        else:
            self.requestli_cache = [{'balance': 0, 'data': 'Default', 'model': None, 'addr': None, 'taskhash': None}]
        self.block.append(self.block_cache[bidx])
        self.block_cache = []
        self.sinOut.emit('4')
        time.sleep(1)
        return

    def getNouce(self):
        return random.random()

    def run(self):
        app = Flask(__name__)

        @app.route('/')
        def index():
            return '<h1>Welcome using PoLe Demo</h1>', 200

        @app.route('/pole/protocol/V1')
        def protocol():
            return jsonify({'port': self.port, 'status': 'working', 'addr': self.addr, 'msg':200, 'nodes': self.node_list, 'blocks': self.block}), 200

        @app.route('/pole/protocol/addnode', methods=['POST'])
        def add_node():
            self.sinOut.emit('1')
            jsondata = json.loads(request.get_data())
            self.node_list.append('{}'.format(jsondata['ipaddr']))
            return jsonify({'msg': 200}), 200

        @app.route('/get-request', methods=['POST'])
        def get_request():
            self.sinOut.emit('2')
            # 数据节点请求
            datanoderequest = json.loads(request.get_data())
            taskhash = datanoderequest['taskhash']
            if taskhash in self.taskhashlist:
                return jsonify({'msg': 200}), 200

            # 加入训练请求队列
            self.requestli_cache.append(datanoderequest)
            # 任务哈希队列
            self.taskhashlist.append(taskhash)
            t1 = threading.Thread(target=broadcastinvf, args=(self.node_list, datanoderequest, self.ipaddr))
            t1.start()

            return jsonify({'msg': 200}), 200

        # 广播给其他共识节点
        def broadcastinvf(nodelist, requestdata, selfip):
            for ip in nodelist:
                try:
                    if ip == selfip:
                        continue
                    requests.post('{}/get-request'.format(ip), json=requestdata)
                except Exception as e:
                    print('[E4]:{}'.format(e))

        @app.route('/explore')
        def explore():
            return jsonify(self.block), 200

        @app.route('/pole/protocol/consensus', methods=['POST'])
        def consensus():
            self.sinOut.emit('3')
            # 保存其他共识节点生成的区块
            self.block_cache.append(json.loads(request.get_data()))
            return jsonify({'msg': 200}), 200

        # 查询当前taskhash的任务是否完成
        @app.route('/datanode/ifdone/<taskhash>', methods=['GET'])
        def datanoderequest(taskhash):
            print(self.block)
            # for idx in range(len(self.block)-1, 0, -1):
            for idx in range(len(self.block)):    
                print(idx)
                if self.block[idx]['donetask']['taskhash'] == taskhash:
                    return jsonify({'code': 200,
                                    'blockidx': idx,
                                    'msg': 'task is done',
                                    'acc': self.acc,
                                    'hash': taskhash,
                                    'content': '<Model><Weight></Weight></Model>'}), 200
            print("nono")
            return jsonify({'code': 500,
                            'msg': 'task wait in queue.'})

        app.run(port=self.port, threaded=True)


class FirstUi(QMainWindow): 
    def __init__(self, port, extnodeli, extblockli):
        super(FirstUi, self).__init__()
        self._echo = ''
        self._count = 2253
        self.r = None
        self.b = 1000
        self.language = 0
        self.port = port
        self.extnodeli = extnodeli
        self.thread = Worker()
        r = hashlib.sha256(bytes(str(random.random()), encoding='utf8')).hexdigest()
        print('self addr: {}'.format(r))
        self.r = r
        self.thread.sinOut.connect(self.interrupt1)
        self.thread.configargs(port=self.port, addr=r, nodes=extnodeli, blocks=extblockli)
        self.thread.start()
        # 共识时间
        self.c1 = 20
        # 训练时间
        self.c2 = 300
        self.countdown1 = self.c1 
        self.countdown2 = self.c2

        self.init_ui()

    def init_ui(self):

        self.resize(800, 400)  
        self.setWindowTitle('Consensus Node') 

        self.lable1 = QLabel('<h1>PoLe Demo</h1>', self)
        self.lable1.setGeometry(330, 20, 300, 25)

        self.lable4 = QLabel('IP : http://127.0.0.1:{}'.format(self.port), self)
        self.lable4.setGeometry(550, 15, 300, 25)

        self.lable5 = QLabel('Console:', self)
        self.lable5.setGeometry(240, 70, 300, 25)

        self.lable6 = QLabel('<h2>Balance:</h2>', self)
        self.lable6.setGeometry(30, 50, 300, 25)

        self.lable7 = QLabel('<h2>Address:</h2>', self)
        self.lable7.setGeometry(30, 200, 300, 25)

        self.lable8 = QLabel('<h1>{}</h1>'.format(self.b), self)
        self.lable8.setGeometry(30, 90, 300, 25)

        self.lable9 = QLabel('<h1>{}</h1>'.format(self.r[:10]), self)
        self.lable9.setGeometry(30, 240, 300, 25)

        self.lable10 = QLabel('<h2>Status:</h2>', self)
        self.lable10.setGeometry(30, 320, 300, 25)

        self.lable11 = QLabel('<h1>Working</h1>', self)
        self.lable11.setGeometry(30, 340, 300, 50)

        self._echo = """        Event                |   Entity  |       Time       |  IP Addr
-----------------------------------------------------------------------    
        """

        self.tb = QTextBrowser(self)
        self.tb.setText(self._echo)
        self.tb.setGeometry(240, 100, 500, 220)

        self.btn5 = QPushButton('Refresh', self)  
        self.btn5.setGeometry(320, 320, 100, 25)  
        self.btn5.clicked.connect(self.slot_btn_function)  

        self.lable2 = QLabel('Status: Collecting Requests...', self)
        self.lable2.setGeometry(450, 320, 300, 25)

        self.timer = QBasicTimer()  
        self.timer.start(1000, self)



    def slot_btn_function(self):

        pass

    def timerEvent(self, event):
        if event.timerId() == self.timer.timerId():

            if self.countdown1 == 0:
                self.countdown1 = self.c1
                self.thread.get_winner()
                self.lable2.setText("Status: Collecting Requests...")
            else:
                self.countdown1 -= 1

            # 开始训练
            if self.countdown2 == self.c2:
                self.thread.train()
            if self.countdown2 == 0:
                self.countdown2 = self.c2
                self.thread.generate_block()
                self.append_echo('Start training model.')
                self.lable2.setText("Status: Training Models...")
            else:
                self.countdown2 -= 1


    def append_echo(self, msg):

        self._echo += """{}      | {}|  {} |  127.0.0.1:{}
-----------------------------------------------------------------------
        """.format(msg, hashlib.md5(bytes(msg+str(time.time()), encoding='utf8')).hexdigest()[:10], time.time(), self.port)
        self.tb.setText(self._echo)

    def interrupt1(self, sig):
        if sig == '1':
            self.append_echo('A new node registered.')

        if sig == '2':
            self.append_echo('Receive new request.')

        if sig == '3':
            self.append_echo('Received new block.')

        if sig == '4':
            self.append_echo('Consensus reached.')
            self.lable2.setText("Status: Collecting Requests...")


        if 'TOKENS' in sig:
            self.append_echo('Became winner.')
            fee = str(sig)
            print('[FEE]:{}'.format(fee))
            self.lable8.setText("<h1>{}</h1>".format(self.b + int(fee.split(':')[-1])))
            print('[NOW FEE]:{}'.format(self.b + int(fee.split(':')[-1])))
            self.b += int(fee.split(':')[-1])

def get_a_port():
    originp = 7000
    FLAG = True
    nodelist = []
    blocklist = []

    while FLAG:
        try:
            req = requests.get('http://127.0.0.1:{}/pole/protocol/V1'.format(originp))
            jsondata = json.loads(req.content)
            nodelist = jsondata['nodes']
            blocklist = jsondata['blocks']
            originp += 1
        except:

            FLAG = False

    return originp, nodelist, blocklist


if __name__ == '__main__': 

    appmain = QApplication(sys.argv)
    w = FirstUi(*get_a_port())  # 将第一和窗口换个名字
    w.show()  # 将第一和窗口换个名字显示出来
    sys.exit(appmain.exec_())  # app.exet_()是指程序一直循环运行直到主窗口被关闭终止进程（如果没有这句话，程序运行时会一闪而过）
