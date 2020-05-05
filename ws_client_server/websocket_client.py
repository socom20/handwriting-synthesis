import websocket
import threading
import sys, os



def on_message(ws, message):
    print('on_message:', message)


def on_error(ws, error):
    print("- Se produjo un ERROR: " + str(error))
    

def on_close(ws):
    print("on_close: ### closed ###")
    self = ws.self
    self.close()
    ws.self.connected = False
    return None
    
    
def on_open(ws):    
    print("- Conectado al Servidor !!!")
    ws.self.connected = True
    if ws.self.password is not None:
        print(' Sending password ...')
        try:
            ws.send( ws.self.password )
        except Exception as e:
            print(e, file=sys.stderr)


def start_new_ws(ws):
    ws.run_forever()
    return None



class ws_client:
    def __init__(self,
                 host='localhost',
                 port=8001,
                 use_ssl=False,
                 on_message_function=None,
                 password=None):

        self.host     = host
        self.port     = port
        self.use_ssl  = use_ssl
        self.ws       = None
        self.password = password

        self.connected = False

        if on_message_function is None:
            self.on_message_function = on_message
        else:
            self.on_message_function = on_message_function

        
        return None

        
    def start(self):
        if self.ws is None:
            websocket.enableTrace(False)
            
            self.ws = websocket.WebSocketApp("ws{}://{}:{}".format('s' if self.use_ssl else '', self.host, self.port),
                                             on_message = self.on_message_function,
                                             on_error = on_error,
                                             on_close = on_close)
            self.ws.on_open = on_open
            self.ws.self    = self

            self.th = threading.Thread(target=start_new_ws, args=(self.ws,))
            self.th.start()
        return None
    

    def close(self):
        if self.ws is not None:
            self.ws.close()
            self.ws = None

        return None
    

    def send(self, msg='Hello !!!'):
        if self.ws is not None and self.connected:
            if type(msg) is str:
                self.ws.send(msg, 1)
            elif type(msg) is bytes:
                self.ws.send(msg, 2)
            else:
                print(' - WARNING send: msg type not supported', file=sys.stderr)
                
        else:
            print(' - WARNING send: ws closed, unable to send msg', file=sys.stderr)

        return None
            


if __name__ == '__main__':
    client = ws_client(password='gpt_model')
    client.start()



        
    

    

