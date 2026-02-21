from flask import Flask, render_template, request
import heapq
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# ---------------------------
# HUFFMAN NODE
# ---------------------------

class HuffmanNode:
    def __init__(self, symbol, prob):
        self.symbol = symbol
        self.prob = prob
        self.left = None
        self.right = None
        
    def __lt__(self, other):
        return self.prob < other.prob


# ---------------------------
# BUILD HUFFMAN TREE
# ---------------------------

def build_tree(prob_dict):
    heap = []
    merge_steps = []

    for char, prob in prob_dict.items():
        heapq.heappush(heap, HuffmanNode(char, prob))

    while len(heap) > 1:
        heap = sorted(heap, key=lambda x: x.prob)
        
        left = heap.pop(0)
        right = heap.pop(0)

        merged = HuffmanNode(None, left.prob + right.prob)
        merged.left = left
        merged.right = right

        merge_steps.append((left.symbol, left.prob,
                            right.symbol, right.prob,
                            merged.prob))

        heap.append(merged)

    return heap[0], merge_steps


# ---------------------------
# GENERATE CODES
# ---------------------------

def generate_codes(node, current_code="", codes=None):
    if codes is None:
        codes = {}

    if node.symbol is not None:
        codes[node.symbol] = current_code
        return

    generate_codes(node.left, current_code + "0", codes)
    generate_codes(node.right, current_code + "1", codes)

    return codes


# ---------------------------
# ENCODE
# ---------------------------

def encode_message(message, codes):
    return "".join(codes[ch] for ch in message)


# ---------------------------
# DECODE
# ---------------------------

def decode_message(encoded, root):
    decoded = ""
    current = root

    for bit in encoded:
        current = current.left if bit == "0" else current.right

        if current.symbol:
            decoded += current.symbol
            current = root

    return decoded


# ---------------------------
# TREE PLOT
# ---------------------------

def draw_tree(node, x=0, y=0, dx=1.5, positions=None, edges=None):
    if positions is None:
        positions = {}
    if edges is None:
        edges = []

    positions[node] = (x, y)

    if node.left:
        edges.append((node, node.left, "0"))
        draw_tree(node.left, x-dx, y-1, dx/2, positions, edges)

    if node.right:
        edges.append((node, node.right, "1"))
        draw_tree(node.right, x+dx, y-1, dx/2, positions, edges)

    return positions, edges


def plot_tree(root):
    positions, edges = draw_tree(root)

    fig, ax = plt.subplots(figsize=(10,6))

    for node, (x,y) in positions.items():
        label = f"{node.symbol}:{round(node.prob,3)}" if node.symbol else round(node.prob,3)
        ax.text(x, y, label,
                ha='center',
                va='center',
                bbox=dict(facecolor='lightblue', edgecolor='black'))

    for parent, child, bit in edges:
        x1,y1 = positions[parent]
        x2,y2 = positions[child]
        ax.plot([x1,x2],[y1,y2],'k-')
        ax.text((x1+x2)/2,(y1+y2)/2,bit,color='red')

    ax.axis('off')
    plt.tight_layout()
    plt.savefig("static/tree.png")
    plt.close()


# ---------------------------
# ROUTES
# ---------------------------

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    letters = request.form['letters'].split(',')
    probs = list(map(float, request.form['probabilities'].split(',')))

    prob_dict = dict(zip(letters, probs))

    root, merge_steps = build_tree(prob_dict)
    codes = generate_codes(root)

    message = request.form.get('message', "")
    encoded = ""
    decoded = ""

    if message:
        encoded = encode_message(message, codes)
        decoded = decode_message(encoded, root)

    entropy = -sum(p * math.log2(p) for p in prob_dict.values())
    avg_length = sum(prob_dict[ch] * len(codes[ch]) for ch in codes)

    plot_tree(root)

    return render_template("index.html",
                           prob_dict=prob_dict,
                           codes=codes,
                           merge_steps=merge_steps,
                           encoded=encoded,
                           decoded=decoded,
                           entropy=round(entropy,4),
                           avg_length=round(avg_length,4))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)