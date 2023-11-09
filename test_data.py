#Contains a demo of generator.py

from generator import Generator


if __name__ == "__main__":

    gen = Generator(11)
    gen.draw_graph("outputs/figure", "graph_11")
    pyg_graph = gen.export_pyg()
