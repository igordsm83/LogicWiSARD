#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 08:07:17 2022

@author: igor
"""
import numpy as np

def gen_lut_grouped (wsrd, INDEX_WIDTH, O_WIDTH, path):
    # LUT v2 #############################################################
    code = 'module wisard_lut\n#(parameter ADDR_WIDTH=%d, INDEX_WIDTH=%d, O_WIDTH=%d)\n' % (wsrd.address_size,INDEX_WIDTH,O_WIDTH)
    code += '(input [ADDR_WIDTH-1:0] addr, input [INDEX_WIDTH-1:0] index, output [O_WIDTH-1:0] out);\n\n'
    code += '\nreg [%d:0] out_v [0:%d];\n\n' % (O_WIDTH-1,len(wsrd.model[wsrd.classes[0]])-1)
    code += '\nassign out = out_v[index];\n\n'      
    
    for r in range(len(wsrd.model[wsrd.classes[0]])):
        # code += '\nreg [%d:0] out%d;\n' % (N_CLASSES-1, r)
        code += 'always @(*) begin\n  case (addr)\n'
        unified_ram = {}
        for c in range (len(wsrd.classes)):
            dict_tmp = wsrd.model[wsrd.classes[c]][r]
            for a in dict_tmp:
                ai = int(a)
                if ai in unified_ram:
                    unified_ram[ai] = unified_ram[ai] | (1<<c)
                else:
                    unified_ram[ai] = 1<<c
        for u in unified_ram:
            u_addr = u
            # u_addr = (u + np.random.randint(0,2**1,1)[0]) % (2**14)
            # u_addr = np.random.randint(0,2**14,1)[0]
            
            # u_addr = "{0:014b}".format(u_addr)
            # code += '    %d\'b%s: out_v[%d] = %d\'d%d;\n' % (wsrd.address_size,u_addr,r, O_WIDTH,unified_ram[u])
            
            code += '    %d\'d%d: out_v[%d] = %d\'d%d;\n' % (wsrd.address_size,u_addr,r, O_WIDTH,unified_ram[u])
            
            
        code += '    default: out_v[%d] = %d\'d0;\n  endcase\nend\n\n' % (r,O_WIDTH)
    


    code += '\nendmodule'
    
    text_file = open(path+"wisard_lut.v", "w")
    text_file.write(code)
    text_file.close()      
    
    

def gen_lut_overgrouped (wsrd, INDEX_WIDTH, O_WIDTH, path):
    # LUT v1 ##############################################################
    code = 'module wisard_lut\n#(parameter ADDR_WIDTH=%d, INDEX_WIDTH=%d, O_WIDTH=%d)\n' % (wsrd.address_size,INDEX_WIDTH,O_WIDTH)
    code += '(input [ADDR_WIDTH-1:0] addr, input [INDEX_WIDTH-1:0] index, output [O_WIDTH-1:0] out);\n\n'
    code += 'wire [%d:0] in;\n\n' % (wsrd.address_size+INDEX_WIDTH-1)
    code += 'assign in = {index, addr};\n\n'
    
    code += 'always @(in) begin\n  case (in)\n'
    
    for r in range(len(wsrd.model[wsrd.classes[0]])):
        unified_ram = {}
        for c in range (len(wsrd.classes)):
            dict_tmp = wsrd.model[wsrd.classes[c]][r]
            for a in dict_tmp:
                ai = int(a)
                if ai in unified_ram:
                    unified_ram[ai] = unified_ram[ai] | (1<<c)
                else:
                    unified_ram[ai] = 1<<c
        for u in unified_ram:
            u_addr = (r<<wsrd.address_size) | u
            code += '    %d\'d%d: out = %d\'d%d;\n' % (wsrd.address_size+INDEX_WIDTH,u_addr,O_WIDTH,unified_ram[u])
            
    code += '    default: out = %d\'d0;\n  endcase\nend\n\nendmodule' % (O_WIDTH)
    
    text_file = open(path+"wisard_lut_overgrouped.v", "w")
    text_file.write(code)
    text_file.close()
    


def gen_lut_gates (wsrd, INDEX_WIDTH, O_WIDTH, path):
    # LUT v3 redesign #####################################################
    code = 'module wisard_lut\n#(parameter ADDR_WIDTH=%d, INDEX_WIDTH=%d, O_WIDTH=%d)\n' % (wsrd.address_size,INDEX_WIDTH,O_WIDTH)
    code += '(input [ADDR_WIDTH-1:0] addr, input [INDEX_WIDTH-1:0] index, output [O_WIDTH-1:0] out);\n\n'
    code += '\nwire [%d:0] out_v [0:%d];\n' % (O_WIDTH-1,len(wsrd.model[wsrd.classes[0]])-1)
    # code += '\nreg [%d:0] out_v [0:%d];\n' % (O_WIDTH-1,len(wsrd.model[wsrd.classes[0]])-1)
    code += 'assign out = out_v[index];\n\n'      
    
    
    for r in range(len(wsrd.model[wsrd.classes[0]])):
        code += '\n// RAM %d\n\n' % (r)
        
        # code += 'always @(*) begin\n  case (in[%d:0])\n' % (wsrd.address_size-1)
        
        unified_ram = {}
        for c in range (len(wsrd.classes)):
            dict_tmp = wsrd.model[wsrd.classes[c]][r]
            for a in dict_tmp:
                ai = int(a)
                if ai in unified_ram:
                    unified_ram[ai] = unified_ram[ai] | (1<<c)
                else:
                    unified_ram[ai] = 1<<c

        included = np.zeros((len(unified_ram)), dtype=int)
                
        i=0
        declarations='wire [%d:0] ' % (O_WIDTH-1)
        all_selected = ''
        # mux_struct = 'always @(*) begin\n'
        or_struct = 'assign out_v[%d] = ' % (r)
        for u in unified_ram:
            # u_addr = u
            # u_addr = (u + np.random.randint(0,2**1,1)[0]) % (2**14)
            # u_addr = np.random.randint(0,2**14,1)[0]
            
            # u_addr = "{0:014b}".format(u_addr)
            # code += '    %d\'b%s: out_v[%d] = %d\'d%d;\n' % (wsrd.address_size,u_addr,r, O_WIDTH,unified_ram[u])
            
            if included[i]==0:
                included[i] = 1
                if i>0:
                    declarations += ','
                    # mux_struct += 'else '
                    or_struct += '|'
                declarations += 'hit_r%d_%d' % (r,unified_ram[u])
                selected = 'assign hit_r%d_%d = addr==%d\'d%d' % (r,unified_ram[u],wsrd.address_size,u)
                # mux_struct += 'if (hit_r%d_%d) out_v[%d] = %d\'d%d;\n' % (r,unified_ram[u], r, O_WIDTH, unified_ram[u])
                or_struct += 'hit_r%d_%d' % (r,unified_ram[u])
                
                j = 0
                for u2 in unified_ram:
                    if unified_ram[u2] == unified_ram[u] and included[j]==0:
                        selected += ' || addr==%d\'d%d' % (wsrd.address_size,u2)                    
                        included[j] = 1
                    j+=1
                selected += ' ? %d\'d%d : %d\'d0;\n' % (O_WIDTH, unified_ram[u], O_WIDTH)

                all_selected += selected
            # code += '    %d\'d%d: out_v[%d] = %d\'d%d;\n' % (wsrd.address_size,u_addr,r, O_WIDTH,unified_ram[u])
            
            i+=1
            
        code+= declarations
        code += ';\n\n'                
        code += all_selected
        code += '\n\n'   
        # code += mux_struct + 'else out_v[%d] = %d\'d0;\nend\n' % (r,O_WIDTH)            
        code += or_struct + ';\n\n'            


    code += '\nendmodule'
    
    text_file = open(path+"wisard_lut_gates.v", "w")
    text_file.write(code)
    text_file.close()       
    
def gen_lut_modules (wsrd, INDEX_WIDTH, O_WIDTH, path): 

    # LUT v4 ##############################################################
    
    code = 'module wisard_lut\n#(parameter ADDR_WIDTH=%d, INDEX_WIDTH=%d, O_WIDTH=%d)\n' % (wsrd.address_size,INDEX_WIDTH,O_WIDTH)
    code += '(input [ADDR_WIDTH-1:0] addr, input [INDEX_WIDTH-1:0] index, output [O_WIDTH-1:0] out);\n\n'
    code += '\nwire [%d:0] out_v [0:%d];\n\n' % (O_WIDTH-1,len(wsrd.model[wsrd.classes[0]])-1)
    code += '\nassign out = out_v[index];\n\n'     
    local_code = ''
    for r in range(len(wsrd.model[wsrd.classes[0]])):
        
        
        unified_ram = {}
        for c in range (len(wsrd.classes)):
            dict_tmp = wsrd.model[wsrd.classes[c]][r]
            for a in dict_tmp:
                ai = int(a)
                if ai in unified_ram:
                    unified_ram[ai] = unified_ram[ai] | (1<<c)
                else:
                    unified_ram[ai] = 1<<c
        
        n_chunks = 4
        chunk_size =  int(np.ceil(float(len(unified_ram))/n_chunks))
        declarations = 'wire [%d:0] ' % (O_WIDTH-1)
        or_reduction = 'assign out_v[%d] = ' % (r)
        instances = ''
        i_chk = 0         
        chk_cnt = -1
        for u in unified_ram:
            
            if i_chk%chunk_size==0:
                chk_cnt+=1
                local_code += '\nmodule ram%d_%d (input [%d:0] in, output reg [%d:0] out_l);\n' % (r,chk_cnt,wsrd.address_size-1,O_WIDTH-1)
                local_code += 'always @(*) begin\n  case (in[%d:0])\n' % (wsrd.address_size-1)
                if i_chk>0:
                    declarations += ', '
                    or_reduction += ' | '
                declarations += 'out_v_r%d_%d' % (r, chk_cnt)
                or_reduction += 'out_v_r%d_%d' % (r, chk_cnt)
                instances += 'ram%d_%d ram%d_%d_u0 (.in(addr[%d:0]), .out_l(out_v_r%d_%d));\n' % (r, chk_cnt,r, chk_cnt,wsrd.address_size-1,r, chk_cnt)
                
            u_addr = u    
            local_code += '    %d\'d%d: out_l = %d\'d%d;\n' % (wsrd.address_size,u_addr, O_WIDTH,unified_ram[u])
            # local_code += '    %d\'d%d: out_v_r%d_%d = %d\'d%d;\n' % (wsrd.address_size,u_addr,r, chk_cnt, O_WIDTH,unified_ram[u])
            
            i_chk+=1
            
            if i_chk%chunk_size==0 or i_chk==len(unified_ram):
                local_code += '    default: out_l = %d\'d0;\n  endcase\nend\n' % (O_WIDTH)
                local_code += 'endmodule\n'
                
            
    
        code += '// RAM %d\n\n' % (r)
        code += declarations+';\n'
        code += instances+'\n'
        code += or_reduction+';\n'
        
    code += '\nendmodule'
    code += local_code
    
    text_file = open(path+"wisard_lut_modules.v", "w")
    text_file.write(code)
    text_file.close()        


    
def gen_lut_ungrouped (wsrd, INDEX_WIDTH, O_WIDTH, path):

    # LUT v5 Non-grouped ##################################################
    code = 'module wisard_lut\n#(parameter ADDR_WIDTH=%d, INDEX_WIDTH=%d, O_WIDTH=%d)\n' % (wsrd.address_size,INDEX_WIDTH,O_WIDTH)
    code += '(input [ADDR_WIDTH-1:0] addr, input [INDEX_WIDTH-1:0] index, output [O_WIDTH-1:0] out);\n\n'
    code += '\nreg [%d:0] out_v [0:%d];\n' % (len(wsrd.model[wsrd.classes[0]])-1,len(wsrd.classes)-1)                
    
    for c in range (len(wsrd.classes)):
        code += 'assign out[%d] = out_v[%d][index];\n' % (c,c)
        
    code += '\n'
    
    for c in range (len(wsrd.classes)):
        for r in range(len(wsrd.model[wsrd.classes[0]])):            
            code += 'always @(*) begin\n  case (addr)\n'             
            dict_tmp = wsrd.model[wsrd.classes[c]][r]
            for a in dict_tmp:
                ai = int(a)
                code += '    %d\'d%d: out_v[%d][%d] = 1\'b1;\n' % (wsrd.address_size,ai,c,r)
            code += '    default: out_v[%d][%d] = 1\'b0;\n  endcase\nend\n' % (c,r)
            
    code += '\nendmodule'
    
    text_file = open(path+"wisard_lut_ungrouped.v", "w")
    text_file.write(code)
    text_file.close()    