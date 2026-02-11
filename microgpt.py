"""
The most atomic way to train and inference a GPT LLM in pure, dependency-free Python.
Differences from GPT-2 are minor: rmsnorm instead of layer norm, no biases, square ReLU instead of GeLU nonlinearity.
The contents of this file is everything algorithmically needed to train a GPT. Everything else is just efficiency.
Art project by @karpathy.
"""

import os       # for os.path.exists
import math     # for math.log, math.exp
import random   # for random.seed, random.choices
import argparse # for argparse.ArgumentParser

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_embd', type=int, default=16, help='Number of channels in the Transformer')
parser.add_argument('--n_layer', type=int, default=1, help='Number of layers in the Transformer')
parser.add_argument('--block_size', type=int, default=8, help='Maximum sequence length')
parser.add_argument('--num_steps', type=int, default=1000, help='Number of training steps')
parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads in the Transformer')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()
random.seed(args.seed)
n_embd, block_size, n_layer, n_head = args.n_embd, args.block_size, args.n_layer, args.n_head
head_dim = n_embd // n_head

# Dataset example: the names dataset (one name per line). rest of the code just assumes docs: list[str]
if not os.path.exists('input.txt'):
    import urllib.request
    urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt', 'input.txt')
with open('input.txt', 'r') as file:
    text = file.read()
docs = [line.strip() for line in text.strip().split('\n') if line.strip()]
random.shuffle(docs)

# Tokenizer: simple character-level tokenization with BOS/EOS tokens
chars = ['<BOS>', '<EOS>'] + sorted(list(set(''.join(docs))))
vocab_size = len(chars)
stoi = { ch:i for i, ch in enumerate(chars) } # string to integer
itos = { i:ch for i, ch in enumerate(chars) } # integer to string
BOS, EOS = stoi['<BOS>'], stoi['<EOS>']
print(f"vocab size: {vocab_size}, num docs: {len(docs)}")

# Autograd engine
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    def __repr__(self): return f"Value(data={self.data}, grad={self.grad})"

# Model parameter initialization
matrix = lambda nout, nin, std=0.02: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd, std=0)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd, std=0)
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# Model architecture
def linear(x, w):
    return [sum(w[o][i] * x[i] for i in range(len(x))) for o in range(len(w))]

def softmax(logits):
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id] # token embedding
    pos_emb = state_dict['wpe'][pos_id % block_size] # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding

    for li in range(n_layer):
        # 1) Multi-head attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        val = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(val)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() ** 2 for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    # project to vocab (weight tying with wte)
    logits = linear(x, state_dict['wte'])
    return logits

# Adam optimizer
learning_rate = args.learning_rate
beta1, beta2, eps_adam = 0.9, 0.95, 1e-8
m = [0.0] * len(params) # first moment
v = [0.0] * len(params) # second moment

# Training loop
for step in range(args.num_steps):

    # Take a single training document, tokenize it, and crop to block_size
    doc = docs[step % len(docs)]
    tokens = [BOS] + [stoi[ch] for ch in doc] + [EOS]
    tokens = tokens[:block_size]

    # Forward pass through the document over time dimension
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    lossf = 0.0
    for pos_id in range(len(tokens) - 1):
        logits = gpt(tokens[pos_id], pos_id, keys, values)
        probs = softmax(logits)
        loss = -probs[tokens[pos_id + 1]].log()
        loss = (1 / (len(tokens) - 1)) * loss # average over sequence length
        loss.backward()
        lossf += loss.data

    # Adam update (optimizer)
    lr_t = learning_rate * (1 - step / args.num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1} / {args.num_steps} | loss {lossf:.4f}")

# Inference: generate 5 samples
print("\n--- generation ---")
for sample_idx in range(5):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    generated = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == EOS:
            break
        generated.append(itos[token_id])
    print(f"sample {sample_idx}: {''.join(generated)}")
