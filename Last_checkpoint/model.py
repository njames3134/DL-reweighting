import torch
import torch.nn as nn

# Use GPU hardware acceleration if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ModuleWithGrad(nn.Module):
    # Custom wrapper of standard PyTorch module. This allows functionality to set
    # your own model parameters that have associated gradients.

    def params(self):
       # Simply get the tensors from all custom model parameters.
       # Akin to nn.Module.parameters()
       for name, param in self.get_custom_model_params(self):
            yield param
    
    def get_custom_model_params(self, curr_module=None, memo=None, prefix=''):
        # Returns iterator for all custom model parameters.
        # Similar to nn.named_buffers

        # Create memory set of modules visited
        if memo is None:
            memo = set()

        # Check if curr_module is either CustomLinear or CustomConv2d (only these classes would have 'params_with_grad')
        if hasattr(curr_module, 'params_with_grad'):
            # Get weight and bias parameters (names and tensors)
            for name, p in curr_module.params_with_grad():
                if p is not None:
                    if p not in memo:
                        memo.add(p)
                        yield prefix + ('.' if prefix else '') + name, p
        else:
            # Get parameters for other modules (LeNetWithGrad, Sequential, ReLU, MaxPool2d, LogSoftmax, etc.)
            for name, p in curr_module._parameters.items():
                if p is not None:
                    if p not in memo:
                        memo.add(p)
                        yield prefix + ('.' if prefix else '') + name, p
        
        # Recursively check submodules for custom model parameters
        for sub_name, submodule in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + sub_name
            for name, p in self.get_custom_model_params(submodule, memo, submodule_prefix):
                yield name, p

    def update_params_SGD_step(self, lr, grad_t):
        # Perform a single SGD step using provided learning rate and gradient
        # This preserves model parameter gradient history unlike the typical optimizer.step()
        for p, grad in zip(self.get_custom_model_params(self), grad_t):
            curr_name, curr_param = p
            # Classic SGD step
            new_param = curr_param - lr * grad
            self.set_param(self, curr_name, new_param)

    def set_param(self, curr_mod, name, param):
        # Recursive function used to set models parameters (tensors) at lowest levels
        # of ModuleWithGrad objects
        # Must be recursive because of the requrired named_children() for loop.

        # If periods in name then there are submodules in curr_mod
        if '.' in name:
            # Get names of all sub modules, including current module
            n = name.split('.')
            module_name = n[0]
            submodule_names = '.'.join(n[1:])
            # Loop through all submodules (named children)
            for sub_name, sub_mod in curr_mod.named_children():
                # If first submodule (we don't want submodules of submodule here)
                if module_name == sub_name:
                    # Recursively call next submodule
                    self.set_param(sub_mod, submodule_names, param)
                    break
        else:
            # If at lowest submodule level we can set the variable
            setattr(curr_mod, name, param)


class CustomLinear(ModuleWithGrad):
    def __init__(self, *args, **kwargs):
        # Initialize parent class (ModuleWithGrad)
        super().__init__()

        # Pass all arguments to standard (base) PyTorch class
        base_class = nn.Linear(*args, **kwargs)
       
        # Store custom weight and bias tensors
        weight_with_grad = base_class.weight.data.to(device).requires_grad_()
        bias_with_grad = base_class.bias.data.to(device).requires_grad_()
        # Note that reigster_buffer is used here to set model parameters which should not be
        # trained by the optimizer. They will be returned by state_dict though, unlike if you
        # use register_parameter.
        self.register_buffer('weight', weight_with_grad)
        self.register_buffer('bias', bias_with_grad)
        
    def forward(self, x):
        # Forward pass through layer
        # Use functional so we can handle weights and biases ourselves
        return nn.functional.linear(x, self.weight, self.bias)
    
    def params_with_grad(self):
        # Returns custom tensors (model parameters) that have an associated gradient function
        return [('weight', self.weight), ('bias', self.bias)]


class CustomConv2d(ModuleWithGrad):
    def __init__(self, *args, **kwargs):
        # Initialize parent class (ModuleWithGrad)
        super().__init__()

        # Pass all arguments to standard (base) PyTorch class
        base_class = nn.Conv2d(*args, **kwargs)
        
        # Store initalized parameters from base_class
        # These parameters will not be optimized
        self.stride = base_class.stride
        self.padding = base_class.padding
        self.dilation = base_class.dilation
        self.groups = base_class.groups
        
        # Store custom weight and bias tensors
        weight_with_grad = base_class.weight.data.to(device).requires_grad_()
        bias_with_grad = base_class.bias.data.to(device).requires_grad_()   # Will crash if bias is None
        # Note that reigster_buffer is used here to set model parameters which should not be
        # trained by the optimizer. They will be returned by state_dict() though, unlike if you
        # use register_parameter.
        self.register_buffer('weight', weight_with_grad)
        self.register_buffer('bias', bias_with_grad)

    def forward(self, x):
        # Forward pass through layer
        # Use functional so we can handle weights and biases ourselves
        return nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
    def params_with_grad(self):
        # Returns custom tensors (model parameters) that have an associated gradient function
        return [('weight', self.weight), ('bias', self.bias)]


class AlexNet(ModuleWithGrad):
    # LeNet5 model with gradient preservation through SGD model parameter updates.
    # Note that custom convolution and linear layers needed to be used here. These
    # are modification of the original PyTorch layers that allow direct setting of the
    # model parameters with gradient. You do not need to rely on optimizer.step()
    # to update model parameters which provides the much needed flexibility.

    def __init__(self):
        # Initialize parent class (ModuleWithGrad)
        super(AlexNet, self).__init__()
    
        # Begin convolutional (main) layers
        layers = []
        # Layer 1: First Convolution
        layers.append(CustomConv2d(3, 64, kernel_size=11, stride=4, padding=2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        
        # Second Convolution
        layers.append(CustomConv2d(64, 192, kernel_size=5, padding=2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        
        # Third Convolution
        layers.append(CustomConv2d(192, 384, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        # Fourth Convolution
        layers.append(CustomConv2d(384, 256, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        # Fifth Convolution
        layers.append(CustomConv2d(256, 256, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        # Compile all convolutional layers
        self.conv_layers = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Begin MLP (dense) layers
        layers = []
        # Layer 6: First Dense MLP Layer
        layers.append(nn.Dropout(p=0.5))
        layers.append(CustomLinear(256 * 6 * 6, 4096))
        layers.append(nn.ReLU())
        
        # Layer 7: Second Dense MLP Layer
        layers.append(nn.Dropout(p=0.5))
        layers.append(CustomLinear(4096, 4096))
        layers.append(nn.ReLU())

        layers.append(CustomLinear(4096, 1))
        # Perform softmax on final 4 classification nodes
        # layers.append(nn.Sigmoid())
        # Compile all dense (fully connected) layers
        self.dense_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        # Forward pass through the model using input data x

        # Run through convolutional layers
        x = self.conv_layers(x)
        

        x = self.avgpool(x)
        # Flatten output from convolutional layers
        # x = x.reshape((-1, 120))
        x = torch.flatten(x, 1)

        # Run through dense layers
        x = self.dense_layers(x)
        
        return x.squeeze()


class LeNet(ModuleWithGrad):
    # LeNet5 model with gradient preservation through SGD model parameter updates.
    # Note that custom convolution and linear layers needed to be used here. These
    # are modification of the original PyTorch layers that allow direct setting of the
    # model parameters with gradient. You do not need to rely on optimizer.step()
    # to update model parameters which provides the much needed flexibility.

    def __init__(self):
        # Initialize parent class (ModuleWithGrad)
        super(LeNet, self).__init__()
    
        # Begin convolutional (main) layers
        layers = []
        # Layer 1: First Convolution
        layers.append(CustomConv2d(3, 6, kernel_size=5))
        layers.append(nn.ReLU())
        # Layer 2: First Pooling
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        # Layer 3: Second Convolution
        layers.append(CustomConv2d(6, 16, kernel_size=5))
        layers.append(nn.ReLU())
        # Layer 4: Second Pooling
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        # Layer 5: Third Convolution
        layers.append(CustomConv2d(16, 120, kernel_size=5))
        layers.append(nn.ReLU())
        # Compile all convolutional layers
        self.conv_layers = nn.Sequential(*layers)
        
        # Begin MLP (dense) layers
        layers = []
        # Layer 6: First Dense MLP Layer
        layers.append(CustomLinear(38880, 84))
        layers.append(nn.ReLU())
        # Layer 7: Second Dense MLP Layer
        layers.append(CustomLinear(84, 1))
        # layers.append(nn.Sigmoid())
        # Perform softmax on final 10 classification nodes
        # layers.append(nn.LogSoftmax(dim=1))
        # Compile all dense (fully connected) layers
        self.dense_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        # Forward pass through the model using input data x

        # Run through convolutional layers
        x = self.conv_layers(x)

        # Flatten output from convolutional layers
        x = x.reshape((x.size(0), -1))

        # Run through dense layers
        x = self.dense_layers(x)
        
        return torch.squeeze(x)


