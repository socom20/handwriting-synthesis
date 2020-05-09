import numpy as np
import os, sys
import pickle
import traceback
import time
import glob

from websocket_client import ws_client

sys.path.append('../')
from writer import plot_cnc_strokes


def on_message(ws, message):
    pred_obj = ws.pred_obj

##    print('on_message:', message)
    try:
        if type(message) in [bytes, bytearray]:
            
            pred_d = pickle.loads(message)

            textfile = pred_d['textfile']
            cncfile  = os.path.splitext(textfile)[0] + '.pickle'
            figfile  = os.path.splitext(textfile)[0] + '.png'

            with open(cncfile, 'wb') as f:
                f.write(message)
                print(' Saved CNC path at:', cncfile)
                
            message_d = pickle.loads(message)
            plot_cnc_strokes([message_d['cnc_generation']],
                             plot_e=False,
                             save_file=figfile,
                             do_show=False)
                
        else:
            print('on_message: type={} msg={}'.format(type(message), message))
            
    except Exception as e:
        print(' - ERROR on message analysis:', e)
        traceback.print_exc()
        
        
    return None


class HWGeneatorClient():
    def __init__(self,
                 generation_dir='../gpt_generations',
                 txt_ext='txt',
                 host='localhost',
                 port=7010,
                 password='gpt_model',):
        
        self.generation_dir = generation_dir
        self.txt_ext = txt_ext
        
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

    def generate(self,
                 textfile='../text_samples/003.txt',
                 style=None,
                 max_line_hight=None,
                 max_line_width=None,
                 max_chars_in_line=None,
                 p_start=np.array([0.0, 0.0]),
                 new_line_prop=1.03,
                 verbose=False):
        
        self.connect()
        
        if self.client.connected:

            with open(textfile, 'r') as f:
                text = f.read()
                
            data_d = {
                'text':text,
                'style':style,
                'textfile':textfile,
                'max_line_hight':max_line_hight,
                'max_line_width':max_line_width,
                'max_chars_in_line':max_chars_in_line,
                'p_start':p_start,
                'new_line_prop':new_line_prop}

            data_bytes = pickle.dumps(data_d)
            self.client.send( data_bytes )

            if verbose:
                print(' Message sent, wait for response ...')
        else:
            raise Exception(' - ERROR, the client is not connected ...')

            
        return None
        

    def generate_all_hw_files(self,
                              style=None,
                              max_line_hight=None,
                              max_line_width=None,
                              max_chars_in_line=None,
                              p_start=np.array([0.0, 0.0]),
                              new_line_prop=1.03):

        files_v = glob.glob(os.path.join(self.generation_dir,'*/*.{}'.format(self.txt_ext)))

        print(' - Found {} files.'.format(len(files_v)))
        for file in files_v:
            hw_file = os.path.splitext(file)[0] + '.pickle'
            if os.path.exists(hw_file):
                print(' - Omitting existing file: "{}"'.format(hw_file))
                continue


            print(' - Generating file: {} '.format(hw_file))
            self.generate(
                textfile=file,
                style=style,
                max_line_hight=max_line_hight,
                max_line_width=max_line_width,
                max_chars_in_line=max_chars_in_line,
                p_start=p_start,
                new_line_prop=new_line_prop,
                verbose=False)
        

        return None

            
if __name__ == '__main__':
    
    hw_generator_client = HWGeneatorClient(
        host='localhost',
        port=7010,
        generation_dir='../../../../gpt_generations')
    
##    hw_generator_client.generate(
##        textfile='../text_samples/003.txt',
##        max_line_hight=None,
##        max_line_width=None,
##        max_chars_in_line=None,
##        p_start=np.array([0.0, 0.0]),
##        new_line_prop=1.03)


    hw_generator_client.generate_all_hw_files(style=0,
                                              max_line_hight=10,
                                              max_line_width=None,
                                              max_chars_in_line=None)



    
