import numpy as np
import os, sys
import pickle
import traceback
import time

from websocket_client import ws_client



def on_message(ws, message):
    pred_obj = ws.pred_obj

##    print('on_message:', message)
    try:
        if type(message) in [bytes, bytearray]:
            
            pred_d = pickle.loads(message)

            textfile = pred_d['textfile']
            cncfile  = os.path.splitext(textfile)[0] + '.cnc'

            with open(cncfile, 'wb') as f:
                f.write(message)

            print(' Saved CNC path at:', cncfile)
                
        else:
            print('on_message: type={} msg={}'.format(type(message), message))
            
    except Exception as e:
        print(' - ERROR on message analysis:', e)
        traceback.print_exc()
        
        
    return None


class HWGeneatorClient():
    def __init__(self,
                 generation_dir='../gpt_generations',
                 host='localhost',
                 port=7010,
                 password='gpt_model'):
        
        self.generation_dir = generation_dir
        
        self.host = host
        self.port = port
        self.password = password


        self.client = ws_client(host=self.host,
                                port=self.port,
                                on_message_function=on_message,
                                password=self.password)

        return None



    def connect(self, timeout=10):
        if not self.client.connected:
            self.client.start()
            self.client.ws.pred_obj = self

            # Wait connection
            dt = 0.5
            t_w = 0
            t_s = time.time()
            while t_w < timeout:
                time.sleep(dt)
                t_w += dt
                if self.client.connected:
                    print(' Connected!!')
                    break
                else:
                    print('.', end='')
                
            if not self.client.connected:
                print(' - WARNING: the client is not connected to the server.')
                
        return None

    def close(self):
        if self.client.connected:
            self.client.close()
        
        return None

    def generate(self, textfile='../text_samples/003.txt', max_line_hight=None, max_shars_in_line=None, p_start=np.array([0.0, 0.0]), new_line_prop=1.03):
        
        self.connect()
        
        if self.client.connected:

            with open(textfile, 'r') as f:
                text = f.read()
                
            data_d = {
                'text':text,
                'textfile':textfile,
                'max_line_hight':max_line_hight,
                'max_shars_in_line':max_shars_in_line,
                'p_start':p_start,
                'new_line_prop':new_line_prop}

            data_bytes = pickle.dumps(data_d)
            self.client.send( data_bytes )

            print(' Message sent, wait for response ...')
        else:
            raise Exception(' - ERROR, the client is not connected ...')

            
        return None
        

    

            
if __name__ == '__main__':
    
    hw_generator_client = HWGeneatorClient()
    
    hw_generator_client.generate(
        textfile='../text_samples/003.txt',
        max_line_hight=None,
        max_shars_in_line=None,
        p_start=np.array([0.0, 0.0]),
        new_line_prop=1.03)



    
