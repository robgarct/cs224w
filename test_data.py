#Contains a demo of generator.py

from generator import Generator


if __name__ == "__main__":

    gen = Generator(5)
    gen.draw_graph("outputs/figure", "graph_11")
    pyg_graph = gen.export_pyg()
    nx_graph = gen.get_graph()
    solver_data = gen.export_solver_data()
    print(solver_data)



