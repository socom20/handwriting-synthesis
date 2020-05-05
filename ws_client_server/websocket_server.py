import sys, os
import io
from PIL import Image

import threading
import signal
import ssl
from SimpleWebSocketServer import WebSocket, SimpleWebSocketServer, SimpleSSLWebSocketServer
from optparse import OptionParser

import numpy as np
import json
import pickle
sys.path.append('../')

from writer import Writer
import traceback


clients = []
class Gender_predictor(WebSocket):

    def __init__(self, *args_v, **args_d):
        global model
        self.model = model
        
        super().__init__(*args_v, **args_d)
        return None
    
    def predict(self, data):
        
        pred_args_d = pickle.loads(data)

        # e.i.: pred_args_d={'raw_text'='Mi querido hijo, no sabes la alegría que me dió leer tu carta', 'nsamples'=1, 'input_lang'='es', 'output_lang'='es', 'gender_dir'='male'}

        resp_d = dict(pred_args_d)

        del(pred_args_d['textfile'])

        print(' Generating HW samples ...')
        cnc_generation = self.model.gen_cnc_strokes_form_text(**pred_args_d)
        print(' Generation OK!')
        
        resp_d['cnc_generation'] = cnc_generation
        resp_bytes = pickle.dumps(resp_d)
        
        return resp_bytes

    
    def handleConnected(self):
        print(self.address, 'connected ...')
        
        for client in clients:
            client.sendMessage(self.address[0] + ' - connected')
            
        clients.append(self)
        return None


    def handleMessage(self):
        global data
        print(' - msg_type={} from={}'.format(type(self.data), self.address))

        try:
            if 'pass_ok' not in dir(self):
                self.pass_ok = False
                
            if self.pass_ok == False:
                if ws_server.password is None:
                    # There is no password
                    self.pass_ok = True
                    
                elif type(self.data) is str:
                    # This msg is forvalidateing the password
                    if self.data == ws_server.password:
                        self.pass_ok = True
                        print(' - handleMessage: PassWord OK!!')
                        
                        e_s = json.dumps({'msg': 'PassWord OK!!'})
                        self.sendMessage( e_s )

                        # Nothing more to do!!!
                        return None
                    
                    else:
                        print(' - handleMessage: BaD PassWord', file=sys.stderr)
                    
        
            if self.pass_ok:
                # Everything is ok, lets interpretate the msg
                if type( self.data ) is bytearray:
                    data = bytes(self.data)
##                    print(' Makeing prediction ...')
                    try:
                        pred = self.predict(data)
                    except Exception as e:
                        traceback.print_exc()
                        
                        e_d = {'error': str(e), 'where':'predict'}
                        e_s = json.dumps(e_d)
                        print("WARNING:", e, file=sys.stderr)
                        self.sendMessage( e_s )
                        return None
                        
##                    print(' - sending prediction ...')
                    self.sendMessage( pred )

                elif type( self.data ) is str:
                    self.sendMessage(' RC:' + self.data)
                
            else:
                # Everything is not ok, connection refused!!!
                # Password rejected
                e_d = {'error': 'critical ;)', 'where':'handleMessage'}
                e_s = json.dumps(e_d)
                self.sendMessage( e_s )
                self.close()


        except Exception as e:
            e_d = {'error': str(e), 'where':'handleMessage: unhandled'}
            e_s = json.dumps(e_d)
            self.sendMessage( e_s )
            print(' - ERROR, handleMessage: unhandled:', e, file=sys.stderr)
    
        return None

    def handleClose(self):
        clients.remove(self)
        print (self.address, 'closed')
        for client in clients:
            client.sendMessage(self.address[0] + u' - disconnected')



def start_new_server(server):
    try:
        server.serveforever()
    except ValueError as e:
        pass
    except Exception as e:
        raise e
    
    return None



class ws_server:
    def __init__(self,
                 ws_class,
                 host='localhost',
                 port=8000,
                 use_ssl=False,
                 certfile='',
                 keyfile='',
                 password='rtypopuioghj951435dsads'):
        
        self.host = host
        self.port = port
        self.ws_class = ws_class
        self.use_ssl  = use_ssl

        self.certfile = certfile
        self.keyfile  = keyfile
        ws_server.password = password


        self.ssl_version = ssl.PROTOCOL_TLSv1

        self.server = None
        
        return None

        
    def start(self):
        if self.server is None:
            if not self.use_ssl:
                self.server = SimpleWebSocketServer(self.host,
                                                    self.port,
                                                    self.ws_class,
                                                    selectInterval=0.1)

            else:
                self.server = SimpleSSLWebSocketServer(self.host,
                                                       self.port,
                                                       self.ws_class,
                                                       self.certfile,
                                                       self.keyfile,
                                                       version=self.ssl_version,
                                                       selectInterval=0.1,
                                                       ssl_context=None)

            
            print(' - Starting WS Server, {}:{}'.format(self.host, self.port))
            
            self.th = threading.Thread(target=start_new_server, args=(self.server,))
            self.th.start()
            
        return None
    

    def close(self):
        if self.server is not None:
            print(' - Closing WS Server ... Bye')
            self.server.close()
            self.server = None


        
if __name__ == "__main__":
    global model

    
    # Parse arguments
    parser = OptionParser(usage="usage: %prog [options]", version="%prog 1.0")
    parser.add_option("--host", default='0.0.0.0', type='string', action="store", dest="host", help="hostname (localhost)")
    parser.add_option("--port", default=8002, type='int', action="store", dest="port", help="port (80)")
    parser.add_option("--password", default=None, type='str', action="store", dest="password", help="server password")
    parser.add_option("--ssl", default=0, type='int', action="store", dest="ssl", help="ssl (1: on, 0: off (default))")
    parser.add_option("--cert", default='./cert.pem', type='string', action="store", dest="cert", help="cert (./cert.pem)")
    parser.add_option("--key", default='./key.pem', type='string', action="store", dest="key", help="key (./key.pem)")
    
    parser.add_option("--checkpoint-dir", default='../checkpoints', type='string', action="store", dest="checkpoint_dir", help="The folder where the checkpoints are.")
    parser.add_option("--bias", default=0.75, type='float', action="store", dest="bias", help="Bias parameter for the generation.")
    parser.add_option("--styles_-dir", default='../styles', type='string', action="store", dest="styles_dir", help="The folder where the styles are.")
    parser.add_option("--style", default=None, type='int', action="store", dest="style", help="A number indicating the type of the generated text.")
    
    (options, args) = parser.parse_args(['--password', 'gpt_model'])

    print('Starting HW Generator ...')
    # Starting Predictor as a object model
        
    model = Writer(
            checkpoint_dir=options.checkpoint_dir,
            bias=options.bias,
            styles_dir=options.styles_dir,
            style=options.style,
            verbose=True)

    print('Ok!!!')

    
    server = ws_server(ws_class=Gender_predictor,
                       host=options.host,
                       port=options.port,
                       password=options.password,
                       use_ssl=options.ssl,
                       certfile=options.cert,
                       keyfile=options.key)
    
    server.start()

    def close_sig_handler(signal, frame):
        server.close()
        sys.exit()
        return None

    signal.signal(signal.SIGINT, close_sig_handler)








