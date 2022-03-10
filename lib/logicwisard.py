#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 06:30:30 2021

@author: Igor Dantas S. Miranda (igordantas@ufrb.edu.br)
"""
from wisard_base import wisard_train, wisard_eval, wisard_find_threshold, wisard_eval_bin
import math
import os
import numpy as np
from hamming import hamming_correction_sample
import wisard_lut_gen as wsd_lut

class logicwisard:
    """
    Wisardlib is a Wisard classifier library
    This implementation can fit data to a Wisard model using dictionary or 
    cache approaches for RAM modeling. Random mapping and Sequential Bleaching
    are adopted.
    
    Attributes
    ----------
    classes: a list of classes string
    ram_type: 'dict' for dictionary-based RAMs or 'cache' for cache-based RAMs
    address_size: Address bit width
    mapping: Array of shuffled indexes that represents the mapping
    model: A dictionary containing the discriminator for each class
    """
    classes = []
    ram_type = 'dict'
    address_size = 4 
    min_bleaching = 1
    max_bleaching = 100
    best_threshold = 1
    binarized = False
    mapping = []
    model = {}
    model_hamm = {}
    model_conv = {}
    
    def __init__(self, classes, address_size, min_bleaching=1, max_bleaching=100, ram_type = 'dict'):
        self.address_size = address_size
        self.min_bleaching = min_bleaching
        self.max_bleaching = max_bleaching
        self.classes = classes
        self.ram_type = ram_type
    
    
    def fit (self, X,Y):
        """
        Fits data to a Wisard model.

        Parameters
        ----------
        X : numpy array
        A 2-dimensional array with samples in the first dimension
        and input features in the second. For example, to run a training with
        1000 vector of 16 bits each X should have (1000,16) shape. The input 
        data must be binary and the array type can have int type. 
        
        Y : numpy array
            An int array whose values correspond to the classes indexes.
        Returns
        -------
        None.
        """

        self.model, self.mapping = wisard_train (X, Y, self.classes, 
                                                 self.address_size)
        
        
    def classify (self, X, hamming=False):
        """
        Runs the trained classifier on X data.

        Parameters
        ----------
        X : numpy array
            The same format as the fit input.

        Returns
        -------
        Y_pred : numpy array
            An int array whose values correspond to the classes indexes.
        """
        if hamming:
            model_arg = self.model_hamm
        else:
            model_arg = self.model
             
        if self.binarized:
            Y_pred = wisard_eval_bin(X, model_arg, self.mapping, self.classes, 
                                 self.address_size, bleaching=self.min_bleaching, 
                                 hamming=hamming)
        else:
            Y_pred = wisard_eval(X, model_arg, self.mapping, self.classes, 
                                 self.address_size, min_bleaching=self.min_bleaching, 
                                 max_bleaching=self.max_bleaching, hamming=hamming)
        return Y_pred
    

    def find_threshold (self, X, Y, hamming=False):
        """
        Find the threshold that provides the best accuracy. It starts with 
        threshold equal to 1 and stops after 5 consecutive falls.

        Parameters
        ----------
        X : numpy array
            The same format as the fit input.

        Returns
        -------
        Y_pred : numpy array
            An int array whose values correspond to the classes indexes.
        """
        best_acc, best_threshold, best_cnt, max_acc, max_threshold, max_cnt = wisard_find_threshold(X, Y, self.model, self.mapping, self.classes, 
                                 self.address_size, min_bleaching=self.min_bleaching, 
                                 max_bleaching=self.max_bleaching, hamming=hamming)
        self.best_threshold = best_threshold
        return best_acc, best_threshold, best_cnt, max_acc, max_threshold, max_cnt
    
    
    def binarize_model(self):
        """
        This function can be used to apply the threshold to all RAM values.
        The RAM positions with 0 in it is deleted from the dictionaries. 
        The final data could be coverted to vector, but it was left as dictionaries
        for compatility with the bleaching models.
        """

        for c in range (len(self.classes)):  
            for r in range(len(self.model[self.classes[c]])):                
                dict_tmp = self.model[self.classes[c]][r]
                for a in dict_tmp:
                    if dict_tmp[a]>=self.best_threshold:
                        dict_tmp[a] = 1
                    else:
                        dict_tmp[a] = 0
                self.model[self.classes[c]][r]  = {key:val for key, val in dict_tmp.items() if val != 0}
        self.min_bleaching = 1
        self.max_bleaching = 1
        self.binarized = True
        

    def gen_hamming_model(self):
        """
        Create bloom model from binarized model.
        """
        self.model_hamm = {}      
                
        
        n_rams = len(self.model[self.classes[0]])
        
                
        for c in range (len(self.classes)):  # classes
            rams_tmp = []
                
            for r in range(n_rams): # rams
                dict_tmp = self.model[self.classes[c]][r]
                # table_tmp = np.empty((0),dtype=int)
                table_tmp = {}
                           
                for a in dict_tmp: # ram's entries    
                    bin_format = '{0:0%db}' % (self.address_size)
                    ab = [int(a) for a in list(bin_format.format(a))]                       
                    a_hamm = hamming_correction_sample(ab, self.address_size)
                    a_hamm_d = a_hamm.dot(1 << np.arange(a_hamm.shape[-1] - 1, -1, -1))
                    if a_hamm_d not in table_tmp:
                        table_tmp[a_hamm_d] = 1
                rams_tmp.append(table_tmp)
                
            self.model_hamm[self.classes[c]] = rams_tmp
        
    def get_mem_info(self):
        """
        Gets the number of words used for RAMs in the recognizers.
        It also gets the maximum value stored in the RAMs to help on memory
        word decision.
        """
        word_cnt = 0
        max_value = 0
        for c in range (len(self.classes)):        
            for r in range(len(self.model[self.classes[c]])):
                word_cnt += len(self.model[self.classes[c]][r])
                vals = self.model[self.classes[c]][r].values()
                if len(vals)>0:
                    max_t = max(vals)
                else:
                    max_t = 0
                if max_t>max_value:
                    max_value = max_t
        return word_cnt, max_value
    
    def get_minterms_info (self):
        """
        Gets the number of words used for throughout recognizers after the
        minterms fusion.
        """        
        total_min = 0
        for r in range(len(self.model[self.classes[0]])):
            unified_ram = {}
            for c in range (len(self.classes)):
                dict_tmp = self.model[self.classes[c]][r]
                for a in dict_tmp:
                    ai = int(a)
                    if ai in unified_ram:
                        unified_ram[ai] = unified_ram[ai] | (1<<c)
                    else:
                        unified_ram[ai] = 1<<c
            for u in unified_ram:
                total_min += 1

        return total_min
    
    
    def export2verilog(self, path, X, Y):
        """
        Exports model to verilog RTL, creates a testbech and exports the data.

        Parameters
        ----------
        path : String
            Directory where the verilog and data files will be placed.
        X : TYPE
            Input data.
        Y : TYPE
            Output data.            

        Returns
        -------
        code : String
            Verilog code.

        """
        
        ## Exporting verilog code
        x_dim = X.shape[1]
        N_INDEXES = int(x_dim/self.address_size)
        INDEX_WIDTH = math.ceil(math.log2(N_INDEXES))
        I_WIDTH = self.address_size + INDEX_WIDTH
        
        O_WIDTH = len(self.classes)
        
        # Mapping
        
        code = 'module wisard_mapping\n'
        code += '#(parameter ADDRESS_WIDTH = %d, INDEX_WIDTH=%d)\n' % (self.address_size, INDEX_WIDTH)
        code += '(input clk,\ninput rst_n,\ninput sink_sop,\ninput sink_valid,\ninput sink_eop,\n'
        code += 'input [ADDRESS_WIDTH-1:0] addr,\ninput [INDEX_WIDTH-1:0] index,\n'
        code += 'output reg source_sop,\noutput source_valid, \noutput reg source_eop,\n'
        code += 'output reg [ADDRESS_WIDTH-1:0] source_addr, \noutput [INDEX_WIDTH-1:0] source_index);\n\n'        
        code += 'localparam N_INDEXES = %d;\n\n' % (N_INDEXES)
        
        text_file = open('../templates/mapping_v_fragment.txt')
        code += text_file.read()
        text_file.close()        
        
        for m in range (len(self.mapping)):
            code += 'assign out_mem_flat[%d] = in_mem_flat[%d];\n' % (m, self.mapping[m])
                
        code += '\nendmodule'
        
        text_file = open(path+"wisard_mapping.v", "w")
        text_file.write(code)
        text_file.close()        
        

        # Generate verilog LUTs
        wsd_lut.gen_lut_grouped (self, INDEX_WIDTH, O_WIDTH, path)
        
        ## Previous attempts for LUT design 
        # wsd_lut.gen_lut_overgrouped (self, I_WIDTH,O_WIDTH, path)
        # wsd_lut.gen_lut_gates (self, I_WIDTH,O_WIDTH, path)
        # wsd_lut.gen_lut_modules (self, I_WIDTH,O_WIDTH, path)
        # wsd_lut.gen_lut_ungrouped (self, I_WIDTH,O_WIDTH, path)
        
        #######################################################################
        
        
        
        ## Transfering template files
        os.system("cp ../templates/*.v "+path)
        os.system("cp ../templates/*.s* "+path)
        
        ## Set testbench parameters
        tb_params = 'localparam ADDRESS_WIDTH = %d;\n' % (int(self.address_size))
        tb_params += 'localparam N_RAMS = %d;\n' % (int(math.ceil(x_dim/self.address_size)))
        tb_params += 'localparam INDEX_WIDTH = %d;\n' % (int(math.ceil(math.log2(x_dim/self.address_size))))
        tb_params += 'localparam N_CLASSES = %d;\n' % (len(self.classes))
        tb_params += 'localparam CLASS_WIDTH = %d;\n' % (int(math.ceil(math.log2(len(self.classes)))))
        tb_params += 'localparam N_INPUTS = %d;\n' % (X.shape[0])
                
        text_file = open(path+'/tb_wisard.v')
        tb_param_v = text_file.read()
        text_file.close()
        tb_param_v = tb_param_v.replace('//__AUTO_PARAMETERS__', tb_params)
        text_file = open(path+'/tb_wisard.v', "w")
        text_file.write(tb_param_v)
        text_file.close()
        
        ## Set wisard parameters
        text_file = open(path+'/wisard.v')
        wsd_param_v = text_file.read()
        text_file.close()
        wsd_param_v = wsd_param_v.replace('__ADDRESS_WIDTH__', str(int(self.address_size)))
        wsd_param_v = wsd_param_v.replace('__INDEX_WIDTH__', str(int(math.ceil(math.log2(x_dim/self.address_size)))))
        wsd_param_v = wsd_param_v.replace('__N_CLASSES__', str(len(self.classes)))
        wsd_param_v = wsd_param_v.replace('__CLASS_WIDTH__', str(int(math.ceil(math.log2(len(self.classes))))))
        text_file = open(path+'/wisard.v', "w")
        text_file.write(wsd_param_v)
        text_file.close()
        
        ## Exporting data
        os.system("rm -rf "+path+"data")
        os.system("mkdir "+path+"data")
        n_samples = X.shape[0]
        X_mapped = X[:,self.mapping]
        # X_mapped = X
        txt_o = ''
        for n in range (n_samples):
            xt = X_mapped[n,:].reshape(-1, self.address_size)
            # xt = np.flip(xt, axis=1) # flip to correct endianess
            xti = xt.dot(1 << np.arange(xt.shape[-1] - 1, -1, -1))         
            txt_i = ''            
            for m in range (len(xti)):
                txt_i += '%08x\n' % (int(xti[m]))

            txt_o += str(int(Y[n]))+'\n'
            fname = "data/in%04d.txt" % (n)    
            text_file = open(path+fname, "w")
            text_file.write(txt_i)
            text_file.close()

        # #### DEBUG ########
        # X_mapped = X[:,self.mapping]
        # txt_o = ''
        # for n in range (n_samples):
        #     xt = X_mapped[n,:].reshape(-1, self.address_size)
        #     #xt = np.flip(xt, axis=1) # flip to correct endianess

        #     xti = xt.dot(1 << np.arange(xt.shape[-1] - 1, -1, -1))
        #     txt_i = ''            
        #     for m in range (len(xti)):
        #         txt_i += '%04x\n' % (int(xti[m]))
        #         # print(int(xti[m]))
            
        #     txt_o += str(int(Y[n]))+'\n'
        #     fname = "data/m_in%04d.txt" % (n)    
        #     text_file = open(path+fname, "w")
        #     text_file.write(txt_i)
        #     text_file.close()
        #########################
        
        fname = "data/y_pred_sw.txt"
        text_file = open(path+fname, "w")
        text_file.write(txt_o)
        text_file.close()
