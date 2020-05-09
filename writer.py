import os, sys
import logging

import numpy as np

import drawing
from rnn import rnn

import matplotlib.pyplot as plt

import pickle
import string


def plot_cnc_strokes(cnc_strokes, plot_e=False, save_file=None, do_show=True):

    plt.close()
    f, ax = plt.subplots(1, figsize=(10,10))
    
    y_0 = 0.0
    for p_m, e_v in cnc_strokes:
        i_s = 0
        
        for i_e in np.argwhere(e_v).T[0]:
            ax.plot(p_m[i_s:i_e,0], y_0 + p_m[i_s:i_e,1], 'r')
            i_s = i_e
        
        ax.plot(p_m[i_s:,0], y_0 + p_m[i_s:,1], 'r')

        if plot_e:
            for i_e in np.argwhere(e_v).T[0]:
                ax.plot(p_m[i_e-1:i_e+1,0], y_0 + p_m[i_e-1:i_e+1,1], 'b')
                
        y_0 -= 1.03*np.abs(p_m[:,1].max() - p_m[:,1].min())

    if save_file is not None:
        f.savefig(save_file)
        print(' Saved HW figure at: "{}"'.format(save_file))

    if do_show:
        plt.show()

    return None


class Writer(object):

    def __init__(self,
                 checkpoint_dir='./checkpoints',
                 bias=0.75, styles_dir='./styles',
                 default_style=None,

                 chars_from='áéíóúñÁÉÍÓÚÑ\t',
                 chars_to='aeiounAEIOUN ',
                 chars_erase='\r',
                 
                 verbose=False):
        
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.default_style  = default_style
        
        self.style          = default_style
        self.styles_dir     = styles_dir
        self.bias           = bias
        self.checkpoint_dir = checkpoint_dir
        self.verbose        = verbose
        

        self.valid_char_set = set(drawing.alphabet)
        
        self.nn = rnn(
            log_dir='logs',
            checkpoint_dir=checkpoint_dir,
            prediction_dir='predictions',
            learning_rates=[.0001, .00005, .00002],
            batch_sizes=[32, 64, 64],
            patiences=[1500, 1000, 500],
            beta1_decays=[.9, .9, .9],
            validation_batch_size=32,
            optimizer='rms',
            num_training_steps=100000,
            warm_start_init_step=17900,
            regularization_constant=0.0,
            keep_prob=1.0,
            enable_parameter_averaging=False,
            min_steps_to_checkpoint=2000,
            log_interval=20,
            logging_level=logging.CRITICAL,
            grad_clip=10,
            lstm_size=400,
            output_mixture_components=20,
            attention_mixture_components=10
        )
        self.nn.restore()

        self._update_char_trans_d(chars_from=chars_from, chars_to=chars_to, chars_erase=chars_erase)
        
        return None
    

    def write(self, lines_v, do_plot=False):
        
        for line_num in range(len(lines_v)):
            line = lines_v[line_num]
            
            if len(line) > drawing.MAX_CHAR_LEN:
                raise ValueError(
                    (
                        "Each line must be at most {} characters. "
                        "Line {} contains: {}"
                    ).format(drawing.MAX_CHAR_LEN, line_num, len(line))
                )

            for char in line:
                if char not in self.valid_char_set:
                    if self.verbose:
                        print("Invalid character {} detected in line {}. Will be removed.".format(char, line_num), file=sys.stderr)
                    lines_v[line_num] = lines_v[line_num].replace(char, '')
                    
        strokes_v = self._predict_strokes(lines_v)
        strokes_v = [self._align_strokes(stroke, do_plot=do_plot) for stroke in strokes_v]
        
        return strokes_v
        

    def _predict_strokes(self, lines_v):
        num_samples = len(lines_v)
        max_tsteps = 40*max([len(i) for i in lines_v])
        biases = [self.bias]*num_samples if self.bias is not None else [0.5]*num_samples

        x_prime = np.zeros([num_samples, 1200, 3])
        x_prime_len = np.zeros([num_samples])
        chars = np.zeros([num_samples, 120])
        chars_len = np.zeros([num_samples])

        if self.style is not None:
            for i, cs in enumerate(lines_v):
                x_p = np.load(os.path.join(self.styles_dir, 'style-{}-strokes.npy'.format(self.style)))
                c_p = np.load(os.path.join(self.styles_dir, 'style-{}-chars.npy'.format(self.style))).tostring().decode('utf-8')

                c_p = str(c_p) + " " + cs
                c_p = drawing.encode_ascii(c_p)
                c_p = np.array(c_p)

                x_prime[i, :len(x_p), :] = x_p
                x_prime_len[i] = len(x_p)
                chars[i, :len(c_p)] = c_p
                chars_len[i] = len(c_p)

        else:
            for i in range(num_samples):
                encoded = drawing.encode_ascii(lines_v[i])
                chars[i, :len(encoded)] = encoded
                chars_len[i] = len(encoded)

        [samples] = self.nn.session.run(
            [self.nn.sampled_sequence],
            feed_dict={
                self.nn.prime: self.style is not None,
                self.nn.x_prime: x_prime,
                self.nn.x_prime_len: x_prime_len,
                self.nn.num_samples: num_samples,
                self.nn.sample_tsteps: max_tsteps,
                self.nn.c: chars,
                self.nn.c_len: chars_len,
                self.nn.bias: biases
            }
        )
        
        strokes_v = [sample[~np.all(sample == 0.0, axis=1)] for sample in samples]
        return strokes_v


    def _align_strokes(
            self,
            offsets,
            align_strokes=True,
            denoise_strokes=True,
            interpolation_factor=None,
            cnc_format=True,
            do_plot=False
    ):
        
        strokes = drawing.offsets_to_coords(offsets)

        if denoise_strokes:
            strokes = drawing.denoise(strokes)

        if interpolation_factor is not None:
            strokes = drawing.interpolate(strokes, factor=interpolation_factor)

        if align_strokes:
            strokes[:, :2] = drawing.align(strokes[:, :2])

        if do_plot:
            fig, ax = plt.subplots(figsize=(12, 3))

            stroke = []
            for x, y, eos in strokes:
                stroke.append((x, y))
                if eos == 1:
                    coords = tuple(zip(*stroke))
                    ax.plot(coords[0], coords[1], 'y')
                    stroke = []
                    
            if stroke:
                coords = tuple(zip(*stroke))
                ax.plot(coords[0], coords[1], 'y')
                stroke = []
                
            
            ax.set_xlim(-50, 600)
            ax.set_ylim(-40, 40)

            ax.set_aspect('equal')
            plt.tick_params(
                axis='both',
                left='off',
                top='off',
                right='off',
                bottom='off',
                labelleft='off',
                labeltop='off',
                labelright='off',
                labelbottom='off'
            )
            plt.show()
            plt.close('all')

        if cnc_format:
            strokes = self._stroke_to_cnc_format(strokes)
        
        return strokes


    def _stroke_to_cnc_format(self, stroke):
        p_m = stroke[:,:2]
        e_v = stroke[:,2]

        e_v    = np.roll(e_v, 1)
        e_v[0] = 0.0

        f = (e_v == 0)
        for i_e in np.argwhere(e_v).T[0]:
            if i_e == e_v.shape[0]-1 or e_v[i_e+1] == 0.0:
                f[i_e] = True
            
        return (p_m[f], e_v[f])


    def join_cnc_strokes(self,
                         cnc_strokes,
                         p_start=np.array([0.0, 0.0]),
                         new_line_prop=1.03,
                         max_line_hight=None,
                         max_line_width=None):
        
        p_start = np.copy(p_start)

        high_v = [np.abs(p_m[:,1].max() - p_m[:,1].min()) for p_m, e_v in cnc_strokes]
        
        high_norm = 1.0 if max_line_hight is None else max_line_hight / max(high_v)
        
        line_jump = max(high_v) * high_norm * new_line_prop


        to_cat_p_m = []
        to_cat_e_v = []
        for i_s, (p_m, e_v) in enumerate(cnc_strokes):
            p_m = high_norm * p_m
            e_v[0] = 1.0

            if max_line_width is not None:
                line_w = np.abs(p_m[:,0].max() - p_m[:,0].min())
                if line_w > max_line_width:
                    w_factor = max_line_width / line_w
                    p_m[:,0] = w_factor * p_m[:,0]
                    
                
            if i_s == 0:
                to_cat_p_m.append( np.copy(p_start[np.newaxis,:]) )
                to_cat_e_v.append( np.array([0.0]) )
                p_start[1] -= p_m[:,1].max()
            
            
            p_start[0] = -p_m[:,0].min()
            to_cat_p_m.append(p_start + p_m)
            to_cat_e_v.append(e_v)

            p_start[1] -= line_jump

        
        p_m = np.concatenate(to_cat_p_m)
        e_v = np.concatenate(to_cat_e_v)

        return [(p_m, e_v)]

    
    def plot_cnc_strokes(self, cnc_strokes, plot_e=False, save_file=None):
        return plot_cnc_strokes(cnc_strokes=cnc_strokes, plot_e=plot_e, save_file=save_file)


    def _update_char_trans_d(self, chars_from='áéíóúñÁÉÍÓÚÑ\t', chars_to='aeiounAEIOUN ', chars_erase='\r'):

        for C, c in zip(string.ascii_lowercase, string.ascii_uppercase):
            if C not in self.valid_char_set and c in self.valid_char_set:
                if C not in chars_from:
                    chars_from += C
                    chars_to += c

            if C in self.valid_char_set and c not in self.valid_char_set:
                if c not in chars_from:
                    chars_from += c
                    chars_to += C

        self.trans_d = str.maketrans(chars_from, chars_to, chars_erase)
        return self.trans_d
        
    
    def gen_cnc_strokes_form_text(self,
                                  text,
                                  style=None,
                                  max_line_hight=None,
                                  max_line_width=None,
                                  max_chars_in_line=None,
                                  p_start=np.array([0.0, 0.0]),
                                  new_line_prop=1.03,
                                  do_plot=False):

        if style is None:
            self.style = self.default_style
        else:
            self.style = style
            
        if max_chars_in_line is None:
            max_chars_in_line = drawing.MAX_CHAR_LEN
        else:
            max_chars_in_line = min(drawing.MAX_CHAR_LEN, max_chars_in_line)

        
        text = text.translate(self.trans_d)
        split_lines_v = text.split('\n')

        lines_v = []
        for line in split_lines_v:
            line = line.strip()
            if len(line) == 0:
                continue
            
            lines_v.append('')
            for word in line.split():
                if len(lines_v[-1]) + len(word) + 1 < max_chars_in_line:
                    if lines_v[-1] != '':
                        lines_v[-1] += ' '
                    lines_v[-1] += word
                    
                else:
                    lines_v.append(word)

        if len(lines_v) == 0:
            return None
    
        cnc_strokes = self.write(lines_v=lines_v)
        cnc_stroke = self.join_cnc_strokes(cnc_strokes,
                                           p_start=p_start,
                                           new_line_prop=new_line_prop,
                                           max_line_hight=max_line_hight,
                                           max_line_width=max_line_width)

        if do_plot:
            self.plot_cnc_strokes(cnc_stroke,
                                  plot_e=False)
            
        return cnc_stroke[0]

    
if __name__ == '__main__':
    hw = Writer(bias=5)
    
##    # usag demo
##    lines_v = [
##        "Hola David, Este es un ejemplo de escritura con TensorFlow",
##        "Sergio Manuel Papadakis"
##    ]
##
##    cnc_strokes = hw.write(lines_v=lines_v)
##    cnc_strokes_1l = hw.join_cnc_strokes(cnc_strokes, max_line_hight=10)
##    
##    hw.plot_cnc_strokes(cnc_strokes_1l)

    text = """Hijo mío, estás lejos y aquí, a mi costado un compañero tuyo ocupa tu lugar. Nada, nada ha cambiado nada de lo que amamos.
            Este amigo ilumina de esperanza tu ausencia otro, por ti, sin duda, se equivoca. Y aquel joven se bebe toda la primavera en una sola copa, en una boca.
            No necesito ya quererte o encontrarte darte los buenos días. Te tropiezo en la calle, marchas a la par mía y acaso me acompañas como nunca lo hacías.
            Eres todos los jóvenes, eres toda la vida he sembrado de hijos la extensión de la tierra y cuando dicen, madre, me ilumino fiel a mi vocación y tu destino.
            Y así, yo encuentro en Yuri, en César o en Enrique al amable testigo y la sabiduría del instinto certero. Resplandece tu sangre de condecoraciones. Tu sangre y sus legítimos blasones.
            Yo venero a los jóvenes, su destino sellado el callado heroísmo de vivir el placer y las lágrimas, los días por venir.
            Yo descifro el tatuaje de los días futuros -ese papiro en blanco- por eso, a todos quiero cantar mi desagravio. ;Oh juventud! ;Oh Nilo de la vida! por todos los excesos bendecida.
            Y has debido irte lejos. Abro los cinco dedos una estrella es mi mano una estrella en tu cielo. ¿Tienes clara conciencia del privilegio que te significa ser joven así, eternamente, por el milagro renovado de la vida?
            ¡Cómo te acercas cuanto más te alejas! Roto el istmo sutil que nos uniera te represento en otro continente asilado bajo otra bandera.
            Nunca habrá de morir tu juventud, tu sueño, tu sonrisa, no habrás de ser posado, éste, aquél, te han de prestar las manos.
            Ya no serás la noche, nunca serás la noche. Yo soy todas las madres y eres todos los hijos. Al amarlos a todos te amo más a ti mismo…
            Ya nunca estarás solo, multiplicado estás por legiones de jóvenes de cuatro latitudes… para siempre jamás."""
        
    cnc_text = hw.gen_cnc_strokes_form_text(text, style=None, max_line_hight=10, max_line_width=None, max_chars_in_line=None, do_plot=True)

    cnc_text_bin = pickle.dumps(cnc_text)

    file_name = 'cnc_text_bin.pickle'
    with open(file_name, 'wb') as f:
        f.write(cnc_text_bin)
        print('Escrito archivo:', file_name)
        
        
