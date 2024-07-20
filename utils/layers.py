#%% Importing necessary packages 
from functools import partial, reduce

import torch
from torch import nn 
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import brevitas.nn as qnn

from .init import random_restrict_fanin
from .utils import fetch_mask_indices, generate_permutation_matrix
from .verilog import    generate_lut_verilog, \
                        generate_neuron_connection_verilog, \
                        layer_connection_verilog, \
                        generate_logicnets_verilog, \
                        generate_register_verilog
from .bench import      generate_lut_bench, \
                        generate_lut_input_string, \
                        sort_to_bench


# TODO: after making sure that the model is working fine, remove all unnecessary printing statements from all the classes below 


# Customized Linear layer (multiplying the weight matrix with the sparsity mask to induce pruning)
def generate_turth_tables(model: nn.Module, verbose: bool = False): 
    ''' 
    This function take the model as an argument and creates truth tables for the designated layers
    arg: 
    Model: the used nn model 
    verbose: this argument is used if the user prefer a more wordy execution of the code then set it to True 
    to print statements that can help keep track of the code execution
    '''

    # Save the current status of the model in its training mode
    training = model.training 
    # Put the model in evaluation mode
    model.eval() 
    # Iterate over the model layers and create the truth tables 
    for name, module in model.named_modules():
        if type(module) == SparseLinearNeq: # Generating truth tables for Linear FC layers 
            if verbose: 
                print(f"Generating truth table for layer {name}")
            module.calculate_truth_tables()
            if verbose: 
                # Print how many neurons for which the tables were created
                print(f"Truth tables generated for {len(module.neuron_truth_tables)} neurons")

        if type(module) == SparseConv1dNeq: # Generating truth tables for convolutional layers
            if verbose:
                print(f"Generating truth table for layer {name}")
            module.calculate_truth_tables_conv()
            if verbose:
                print(f"Truth tables generated for {len(module.neuron_truth_tables)} neurons") 
    
    # Restore the original traububg mode 
    model.training = training

# examine these two function and what do they exactly do 
# function 1
def lut_inference(model: nn.Module) -> None: 
    for name, module in model.named_modules: 
        if type(module) == SparseLinearNeq: 
            module.lut_inference()

        if  type(module) == SparseConv1dNeq:
            module.lut_inference()

# function 2 
def neq_inference(model: nn.Module) -> None: 
    for name, module in model.named_modules():
        if type(module) == SparseLinearNeq: 
            module.lut_inference()

        if  type(module) == SparseConv1dNeq:
            module.lut_inference()

# the function belos is to transform the model into verilog module. (conv layers are to be added into the function) 
def module_list_to_verilog_module(module_list: nn.ModuleList, module_name: str, output_directory: str, add_registers: bool = True, generate_bench: bool =True): 
    input_bitwidth = None 
    output_bitwidth = None
    module_contents = ''

    for i in range(len(module_list)):
        m = module_list[i]
        if type(m) == SparseLinearNeq: 
            module_prefix = f"layer{i}"
            module_input_bits, module_output_bits = m.gen_layer_verilog(module_prefix, output_directory, generate_bench=generate_bench)
            if i == 0:
                input_bitwidth = module_input_bits
            if i == len(module_list)-1: 
                output_bitwidth = module_output_bits 
            
            module_contents += layer_connection_verilog( module_prefix, 
                                                        input_string = f'M{i}', 
                                                        input_bits = module_input_bits, 
                                                        output_string = f'M{i+1}',
                                                        output_bits = module_output_bits,
                                                        output_wire = i !=len(module_list)-1, 
                                                        register = add_registers) 
        
        else:  
            raise Exception(f'Expect type(module) == SparseLinearNeq, {type(m)} found') 
        
    module_list_verilog = generate_logicnets_verilog( module_name = module_name, 
                                                     input_name = 'M0', 
                                                     input_bits = input_bitwidth, 
                                                     outpuut_name = f'M{len(module_list)}', 
                                                     output_bits = output_bitwidth, 
                                                     module_contents = module_contents)
    
    reg_verilog = generate_register_verilog() 
    with open(f"{output_directory}/myreg.v", 'w') as f: 
        f.write(reg_verilog)

    with open(f'{output_directory}/{module_name}.v', 'w') as f: 
        f.write(module_list_verilog) 

       
# the full SparselinearNeq function: 
class SparseLinear(qnn.QuantLinear): 
    ''' 
    This function induces pruning in the layer by multiplying the weight matrix with the mask matrix
    '''
    def __init__(self, in_features: int, out_features: int, mask: nn.Module, bias: bool = False) -> None:
        super(SparseLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.mask = mask

    def forward(self, input: Tensor) -> torch.Tensor:
        return F.linear(input, self.weight * self.mask(), self.bias)

''' 
The class below includes all the functions as shown in the LogicNets repo for the linear layer, the conv layers are to be edited to include all the functions below
1- the __init__() function introduces some of the variables and also perfroms the SparseLinear function
2- lut_cost() function calculates the number of luts that are needed within every layer, this function overestimates the required LUT number
3- gen_layer_verilog() function calculates the layer contents (# of input and output bits) and then writes the neuron_verilog into directory, also it 
    calucalte various variables to connect sunapses to neurons
4- gen_neuron_verilog() function generates the lut_verilg for neurons
5- gen_neuron_bench() generates lut_bench for the input bitwidth and output bitwidth
6- table_lookup() function creates bin_output_states for the input permutation matrix and will be later used in the lut_forward pass
7- calculate_truth_tables() generates the truth table 
'''
class SparseLinearNeq(nn.Module):
    def __init__(self, in_features: int, out_features: int, input_quant, output_quant, mask, apply_input_quant=True, apply_output_quant=True, first_linear=True, bias = False) -> None:
        super(SparseLinearNeq, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_quant = input_quant
        self.fc = SparseLinear(in_features, out_features, mask, bias)
        self.output_quant = output_quant
        self.is_lut_inference = False
        self.neuron_truth_tables = None
        self.apply_input_quant = apply_input_quant
        self.apply_output_quant = apply_output_quant
        self.first_linear = first_linear


    def lut_cost(self):
        """
        Approximate how many 6:1 LUTs are needed to implement this layer using 
        LUTCost() as defined in LogicNets paper FPL'20:
            LUTCost(X, Y) = (Y / 3) * (2^(X - 4) - (-1)^X)
        where:
        * X: input fanin bits
        * Y: output bits 
        LUTCost() estimates how many LUTs are needed to implement 1 neuron, so 
        we then multiply LUTCost() by the number of neurons to get the total 
        number of LUTs needed.
        NOTE: This function (over)estimates how many 6:1 LUTs are needed to implement
        this layer b/c it assumes every neuron is connected to the next layer 
        since we do not have the next layer's sparsity information.
        """
        # Compute LUTCost of 1 neuron
        _, input_bitwidth = self.input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.output_quant.get_scale_factor_bits()
        input_bitwidth, output_bitwidth = int(input_bitwidth), int(output_bitwidth)
        x = input_bitwidth * self.fc.mask.fan_in # neuron input fanin
        y = output_bitwidth 
        neuron_lut_cost = (y / 3) * ((2 ** (x - 4)) - ((-1) ** x))
        # Compute total LUTCost
        return self.out_features * neuron_lut_cost
    

    def gen_layer_verilog(self, module_prefix, directory, generate_bench: bool = True): 

        _, input_bitwidth = self.input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.output_quant.get_scale_factor_bits()
        input_bitwidth, output_bitwidth = int(input_bitwidth), int(output_bitwidth)
        total_input_bits = self.in_features * input_bitwidth
        total_output_bits = self.out_features * output_bitwidth

        # The line below taked the module_prefix which is a (layer number or name) and print the input and output bitwidth
        layer_contents = f"module {module_prefix} (input [{total_input_bits-1}:0] M0, output[{total_output_bits-1}:0] M1); \n\n"
        output_offset = 0 

        for index in range(self.out_features): 
            module_name = f"{module_prefix}_N{index}" 
            indices, _, _, _ = self.neuron_truth_tables[index]
            neuron_verilog = self.gen_neuron_verilog(index, module_name) 

            with open(f"{directory}/{module_name}.v", "w") as f: 
                f.write(neuron_verilog)
            
            if generate_bench: 
                # Generate the contents of the neuron verilog
                neuron_bench = self.get_neuron_bench(index, module_name)
                with open(f"{directory}/{module_name}.bench", "w") as f: 
                    f.write(neuron_bench) 

            # Generate the string which connects the synapses to this neuron
            connection_string = generate_neuron_connection_verilog(indices, input_bitwidth)
            wire_name = f"{module_name}_wire" 
            connection_line = f"wire [{len(indices)* input_bitwidth-1}:0] {wire_name} = {{{connection_string}}}; \n" 
            inst_line = f"{module_name} {module_name}_inst (.M0({wire_name}), .M1(M1[{output_offset+output_bitwidth-1}:{output_offset}])); \n\n"
            layer_contents += connection_line + inst_line
            output_offset += output_bitwidth
        layer_contents += 'endmodule'

        with open(f'{directory}/{module_prefix}.v', 'w') as f: 
            f.write(layer_contents)
        
        return total_input_bits, total_output_bits


    def gen_neuron_verilog(self, index, module_name): 
        indices, input_perm_matrix, float_output_states, bin_output_states = self.neuron_truth_tables[index] 
        _, input_bitwidth = self.input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.output_quant.get_scale_factor_bits()
        cat_input_bitwidth = len(indices) * input_bitwidth
        lut_string = '' 
        num_entries = input_perm_matrix.shape[0] 
        
        for i in range(num_entries): 
            entry_string = '' 
            for idx in range(len(indices)): 
                val = input_perm_matrix[i, idx]
                entry_string += self.input_quant.get_bin_str(val) 
            
            res_str = self.output_quant.get_bin_str(bin_output_states[i])
            lut_string += f"\t\t\t{int(cat_input_bitwidth)}'b{entry_string}:M1r = {int(output_bitwidth)}'b{res_str};\n"
        return generate_lut_verilog(module_name, int(cat_input_bitwidth), int(output_bitwidth), lut_string)
    

    def gen_neuron_bench(self, index, module_name): 
        indices, input_perm_matrix, float_output_states, bin_output_states = self.neuron_truth_tables[index] 
        _, input_bitwidth = self.input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.output_quant.get_scale_factor_bits()
        cat_input_bitwidth = len(indices) * input_bitwidth
        lut_string = '' 
        num_entries = input_perm_matrix.shape[0]

        # sorting the input perm matrix to match the bench format 
        input_state_space_bin_str = list(map(lambda y: list(map(lambda z:self.input_quant.get_bin_Str(z), y)), input_perm_matrix))
        sorted_bin_output_states = sort_to_bench(input_state_space_bin_str, bin_output_states) 

        # Generate the LUT for each output: 
        for i in range(int(output_bitwidth)): 
            lut_string += f"M1[{i}]             =LUT 0x"
            output_bin_str = reduce(lambda b,c: b+c, map(lambda a:self.output_quant.get_bin_str(a)[int(output_bitwidth)-1-i], sorted_bin_output_states))
            lut_hex_string = f"int{int(output_bin_str, 2):0{int(num_entries/4)}x} "
            lut_string += lut_hex_string 
            lut_string += generate_lut_input_string(int(cat_input_bitwidth))
        
        return generate_lut_bench(int(cat_input_bitwidth), int(output_bitwidth), lut_string) 
    

    def lut_inference(self): 
        self.is_lut_inference = True
        self.input_quant.bin_output()
        self.output_quant.bin_output()


    def neq_inference(self): 
        self.is_lut_inference = False
        self.input_quant.float_output()
        self.output_quant.float_output()


    def table_lookup(self, connected_input: Tensor, input_perm_matrix: Tensor, bin_output_states: Tensor) -> Tensor: 
        fan_in_size = connected_input.shape[1]
        ci_bcast = connected_input.unsqueeze(2) 
        pm_bcast = input_perm_matrix.t().unsqueeze(0) 
        eq = (ci_bcast == pm_bcast).sum(dim=1) == fan_in_size # Create a boolean matrix which matches input vectors to possible input states
        matches = eq.sum(dim=1) # Count the number of perfect matches per input vector

        if not(matches == torch.ones.like(matches, dtype = matches.dtype)).all(): 
            raise Exception(f'One or more vectors in the input is not in the possible input state space')
        indices = torch.argmax(eq.type(torch.int64), dim=1) 
        
        return bin_output_states[indices]
    

    def lut_forward(self, x: Tensor) -> Tensor: 
        if self.apply_input_quant: 
            x = self.input_quant(x) 
        y = torch.zeros(x.shape[0], self.out_features)

        # perfroming table_lookup for each neuron output: 
        for i in range(self.out_features):
            indices, input_perm_matrix, float_output_states, bin_output_states = self.neuron_truth_tables[i] 
            connected_input = x[:,indices]
            y[:,i] = self.table_lookup(connected_input, input_perm_matrix, bin_output_states)

        return y
    

    def forward(self, x: Tensor) -> Tensor:
        if self.is_lut_inference: 
            x = self.lut_forward()
        else: 
            if self.apply_input_quant:
                x = self.input_quant(x)
            if self.first_linear:
                x = x.view(x.size(0), -1)
            x = self.fc(x)
            if self.apply_output_quant and self.output_quant is not None:
                x = self.output_quant(x)
        return x
    

    def calculate_truth_tables(self): 
        with torch.no_grad(): 
            mask = self.fc.mask() 

            # Pre-calculate all of the input value permutations
            input_state_space = list()
            bin_state_space = list() 

            for m in range(self.in_features): 
                print(m)
                neuron_state_space = self.input_quant.get_state_space()
                bin_space = self.input_quant.get_bin_state_space() 
                
                input_state_space.append(neuron_state_space)
                bin_state_space.append(bin_space)
            
            neuron_truth_tables = list() 

            for n in range(self.out_features): 
                # Determine the fan-in as number of synapse connections
                input_mask = mask[n,:]
                fan_in = torch.sum(input_mask) 
                indices = fetch_mask_indices(input_mask)

                # Retrieve the possible state space of the current neuron
                connected_state_space = [input_state_space[i] for i in indices]         
                bin_connected_state_space = [bin_state_space[i] for i in indices] 

                # Generate a matrix containing all possible input states 
                input_permutation_matrix = generate_permutation_matrix(connected_state_space) 
                bin_input_permutation_matrix = generate_permutation_matrix(bin_connected_state_space) 

                num_permutations = input_permutation_matrix.shape[0]
                padded_perm_matrix = torch.zeros((num_permutations, self.in_features))
                padded_perm_matrix[:, indices] = input_permutation_matrix
                print('num_perm: ', num_permutations)
                print('shape perm: ', input_permutation_matrix.shape)

                apply_input_quant, apply_output_quant = self.apply_input_quant, self.apply_output_quant
                self.apply_input_quant, self.apply_output_quant = False, False

                is_bin_output = self.output_quant.is_bin_output 
                self.output_quant.float_output() 
                output_states = self.output_quant(self.forward(padded_perm_matrix))[:, n] # Calculate float for the current input
                print(output_states)
                self.output_quant.bin_output()
                bin_output_states = self.output_quant(self.forward(padded_perm_matrix))[:, n] # Calculate bin for the current input 

                self.output_quant.is_bin_output = is_bin_output 
                self.apply_input_quant, self.apply_output_quant = apply_input_quant, apply_output_quant

                # Append the connectivity, input permutations and output permutations to the neuron truth tables 
                neuron_truth_tables.append((indices, bin_input_permutation_matrix, output_states, bin_output_states))
            
            self.neuron_truth_tables = neuron_truth_tables
    


# Customized convolutional forward function where the weight matrix is multiplied by the mask to induce sparsity into the layer 
class SparseConv1d(qnn.QuantConv1d):
    def __init__(self, in_channels, out_channels, mask:nn.Module, kernel_size=3, padding=1, bias=False) -> None:
        super(SparseConv1d, self).__init__(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.mask = mask
    def forward(self, input) -> Tensor:
        #print('Input shape:', input.shape)
        #print('Weight shape:', self.weight.shape)
        masked_weights = self.weight * self.mask()
        #print('Masked weights shape:', masked_weights.shape)
        output = F.conv1d(input, masked_weights, self.bias, padding=self.padding)
        #print('Output shape of SparseConv1d:', output.shape)
        return output


# Applying the customized convolutional forward function defined above along with the input and/or output quantization function from the brevitas module 
# TODO: this class is to be further customized to allow for more functionality when the hardware is intoduced 
class SparseConv1dNeq(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,  input_quant, output_quant, mask, apply_input_quant=True, apply_output_quant=True, padding=1) -> None:
        super(SparseConv1dNeq, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_quant = input_quant
        #self.mask = mask
        self.padding = padding
        self.conv = SparseConv1d(in_channels, out_channels, mask, kernel_size, padding=padding, bias=False)
        self.output_quant = output_quant
        self.is_lut_inference = False
        self.apply_input_quant = apply_input_quant
        self.apply_output_quant = apply_output_quant 
    
    def lut_cost(self):
        """
        Approximate how many 6:1 LUTs are needed to implement this layer using 
        LUTCost() as defined in LogicNets paper FPL'20:
            LUTCost(X, Y) = (Y / 3) * (2^(X - 4) - (-1)^X)
        where:
        * X: input fanin bits
        * Y: output bits 
        LUTCost() estimates how many LUTs are needed to implement 1 neuron, so 
        we then multiply LUTCost() by the number of neurons to get the total 
        number of LUTs needed.
        NOTE: This function (over)estimates how many 6:1 LUTs are needed to implement
        this layer b/c it assumes every neuron is connected to the next layer 
        since we do not have the next layer's sparsity information.
        """
        # Compute LUTCost of 1 neuron
        _, input_bitwidth = self.input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.output_quant.get_scale_factor_bits()
        input_bitwidth, output_bitwidth = int(input_bitwidth), int(output_bitwidth)
        x = input_bitwidth * self.conv.mask.fan_in # neuron input fanin
        y = output_bitwidth 
        neuron_lut_cost = (y / 3) * ((2 ** (x - 4)) - ((-1) ** x))
        # Compute total LUTCost
        return self.out_channels * neuron_lut_cost
    

    def gen_layer_verilog(self, module_prefix, directory, generate_bench: bool = True): 

        _, input_bitwidth = self.input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.output_quant.get_scale_factor_bits()
        input_bitwidth, output_bitwidth = int(input_bitwidth), int(output_bitwidth)
        total_input_bits = self.in_features * input_bitwidth
        total_output_bits = self.out_features * output_bitwidth

        # The line below taked the module_prefix which is a (layer number or name) and print the input and output bitwidth
        layer_contents = f"module {module_prefix} (input [{total_input_bits-1}:0] M0, output[{total_output_bits-1}:0] M1); \n\n"
        output_offset = 0 

        for index in range(self.out_features): 
            module_name = f"{module_prefix}_N{index}" 
            indices, _, _, _ = self.neuron_truth_tables[index]
            neuron_verilog = self.gen_neuron_verilog(index, module_name) 

            with open(f"{directory}/{module_name}.v", "w") as f: 
                f.write(neuron_verilog)
            
            if generate_bench: 
                # Generate the contents of the neuron verilog
                neuron_bench = self.get_neuron_bench(index, module_name)
                with open(f"{directory}/{module_name}.bench", "w") as f: 
                    f.write(neuron_bench) 

            # Generate the string which connects the synapses to this neuron
            connection_string = generate_neuron_connection_verilog(indices, input_bitwidth)
            wire_name = f"{module_name}_wire" 
            connection_line = f"wire [{len(indices)* input_bitwidth-1}:0] {wire_name} = {{{connection_string}}}; \n" 
            inst_line = f"{module_name} {module_name}_inst (.M0({wire_name}), .M1(M1[{output_offset+output_bitwidth-1}:{output_offset}])); \n\n"
            layer_contents += connection_line + inst_line
            output_offset += output_bitwidth
        layer_contents += 'endmodule'

        with open(f'{directory}/{module_prefix}.v', 'w') as f: 
            f.write(layer_contents)
        
        return total_input_bits, total_output_bits
    
    def gen_neuron_verilog(self, index, module_name): 
        indices, input_perm_matrix, float_output_states, bin_output_states = self.neuron_truth_tables[index] 
        _, input_bitwidth = self.input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.output_quant.get_scale_factor_bits()
        cat_input_bitwidth = len(indices) * input_bitwidth
        lut_string = '' 
        num_entries = input_perm_matrix.shape[0] 
        
        for i in range(num_entries): 
            entry_string = '' 
            for idx in range(len(indices)): 
                val = input_perm_matrix[i, idx]
                entry_string += self.input_quant.get_bin_str(val) 
            
            res_str = self.output_quant.get_bin_str(bin_output_states[i])
            lut_string += f"\t\t\t{int(cat_input_bitwidth)}'b{entry_string}:M1r = {int(output_bitwidth)}'b{res_str};\n"

        return generate_lut_verilog(module_name, int(cat_input_bitwidth), int(output_bitwidth), lut_string)
    
    def gen_neuron_bench(self, index, module_name): 
        indices, input_perm_matrix, float_output_states, bin_output_states = self.neuron_truth_tables[index] 
        _, input_bitwidth = self.input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.output_quant.get_scale_factor_bits()
        cat_input_bitwidth = len(indices) * input_bitwidth
        lut_string = '' 
        num_entries = input_perm_matrix.shape[0]

        # sorting the input perm matrix to match the bench format 
        input_state_space_bin_str = list(map(lambda y: list(map(lambda z:self.input_quant.get_bin_Str(z), y)), input_perm_matrix))
        sorted_bin_output_states = sort_to_bench(input_state_space_bin_str, bin_output_states) 

        # Generate the LUT for each output: 
        for i in range(int(output_bitwidth)): 
            lut_string += f"M1[{i}]             =LUT 0x"
            output_bin_str = reduce(lambda b,c: b+c, map(lambda a:self.output_quant.get_bin_str(a)[int(output_bitwidth)-1-i], sorted_bin_output_states))
            lut_hex_string = f"int{int(output_bin_str, 2):0{int(num_entries/4)}x} "
            lut_string += lut_hex_string 
            lut_string += generate_lut_input_string(int(cat_input_bitwidth))
        
        return generate_lut_bench(int(cat_input_bitwidth), int(output_bitwidth), lut_string)
    
    def lut_inference(self): 
        self.is_lut_inference = True
        self.input_quant.bin_output()
        self.output_quant.bin_output()


    def neq_inference(self): 
        self.is_lut_inference = False
        self.input_quant.float_output()
        self.output_quant.float_output()


    def table_lookup(self, connected_input: Tensor, input_perm_matrix: Tensor, bin_output_states: Tensor) -> Tensor: 
        fan_in_size = connected_input.shape[1]
        ci_bcast = connected_input.unsqueeze(2) 
        pm_bcast = input_perm_matrix.t().unsqueeze(0) 
        eq = (ci_bcast == pm_bcast).sum(dim=1) == fan_in_size # Create a boolean matrix which matches input vectors to possible input states
        matches = eq.sum(dim=1) # Count the number of perfect matches per input vector

        if not(matches == torch.ones.like(matches, dtype = matches.dtype)).all(): 
            raise Exception(f'One or more vectors in the input is not in the possible input state space')
        indices = torch.argmax(eq.type(torch.int64), dim=1) 
        
        return bin_output_states[indices]
    

    def lut_forward(self, x: Tensor) -> Tensor: 
        if self.apply_input_quant: 
            x = self.input_quant(x) 
        y = torch.zeros(x.shape[0], self.out_features)

        # perfroming table_lookup for each neuron output: 
        for i in range(self.out_features):
            indices, input_perm_matrix, float_output_states, bin_output_states = self.neuron_truth_tables[i] 
            connected_input = x[:,indices]
            y[:,i] = self.table_lookup(connected_input, input_perm_matrix, bin_output_states)

        return y

    def forward(self, x: Tensor) -> Tensor:
        if self.is_lut_inference:
            x = self.lut_forward()

        else:     
            if self.apply_input_quant:
                x = self.input_quant(x)
                
            x = self.conv(x)

            if self.apply_output_quant:
                x = self.output_quant(x)

        return x
    
    # This function has been modified to accomodate for the different data shape and behaviour of conv layers 
    def calculate_truth_tables_conv(self):
        with torch.no_grad():
            mask = self.conv.mask()  # Adjust to the appropriate mask for conv layer

            input_state_space = list()
            bin_state_space = list()

            for m in range(self.in_channels):  # iterate over in_channels
                print(m) # can be later removd 
                neuron_state_space = self.input_quant.get_state_space()
                bin_space = self.input_quant.get_bin_state_space()

                input_state_space.append(neuron_state_space)
                bin_state_space.append(bin_space)

            neuron_truth_tables = list()

            for n in range(self.out_channels):  # iterate over out_channels
                for kernel_idx in range(self.kernel_size):  # Iterate over kernel positions

                    # Determine the fan-in as number of synapse connections
                    input_mask = mask[n, :, kernel_idx]
                    fan_in = torch.sum(input_mask)
                    print(fan_in)
                    indices = fetch_mask_indices(input_mask)

                    # Retrieve the possible state space of the current neuron
                    connected_state_space = [input_state_space[i] for i in indices]
                    print('connected state space: ', len(connected_state_space))
                    bin_connected_state_space = [bin_state_space[i] for i in indices]

                    # Generate a matrix containing input states
                    input_permutation_matrix = generate_permutation_matrix(connected_state_space)
                    bin_input_permutation_matrix = generate_permutation_matrix(bin_connected_state_space)

                    num_permutations = input_permutation_matrix.shape[0]
                    padded_perm_matrix = torch.zeros((num_permutations, self.in_channels, self.kernel_size))
                    for idx, perm in zip(indices, input_permutation_matrix.T):
                        padded_perm_matrix[:, idx, kernel_idx] = perm
                    print('num_perm: ', num_permutations)
                    print('shape perm: ', input_permutation_matrix.shape)

                    apply_input_quant, apply_output_quant = self.apply_input_quant, self.apply_output_quant
                    self.apply_input_quant, self.apply_output_quant = False, False

                    is_bin_output = self.output_quant.is_bin_output
                    self.output_quant.float_output()
                    output_states = self.output_quant(self.forward(padded_perm_matrix))[:, n]  # Adjust the forward pass
                    print(output_states)
                    self.output_quant.bin_output()
                    bin_output_states = self.output_quant(self.forward(padded_perm_matrix))[:, n]  # Adjust the forward pass

                    self.output_quant.is_bin_output = is_bin_output
                    self.apply_input_quant, self.apply_output_quant = apply_input_quant, apply_output_quant

                    # Append the connectivity, input permutations and output permutations to the neuron truth tables
                    neuron_truth_tables.append((indices, bin_input_permutation_matrix, output_states, bin_output_states))

            self.neuron_truth_tables = neuron_truth_tables

            
# %%
