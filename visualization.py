import matplotlib.pyplot as plt
import numpy as np
import folium


def heatmap(m, labels, title=''):
    plt.imshow(m, cmap='RdYlGn')
    plt.xticks(list(range(len(labels))), labels, rotation=45)
    plt.yticks(list(range(len(labels))), labels)
    plt.title(title)
    plt.colorbar()
    plt.show()


def draw_path(loc, Pmat, init):
    Pmatc = Pmat.copy()
    path = [init]
    k = 0
    current = np.argmax(Pmatc[init,:])
    while k < (Pmat.shape[0]):
        path.append(current)
        current = np.argmax(Pmatc[current,:])
        Pmatc[:, current] = 0
        k += 1
    path.append(init)

    x = [loc[i][0] for i in path]
    y = [loc[i][1] for i in path]
    plt.plot(x, y, '-o')
    plt.title('Best path')
    plt.show()


def create_map(cities, loc, Pmat, init):

    # compute path
    Pmatc = Pmat.copy()
    path = [init]
    k = 0
    current = np.argmax(Pmatc[init,:])
    while k < (Pmat.shape[0]):
        path.append(current)
        current = np.argmax(Pmatc[current,:])
        Pmatc[:, current] = 0
        k += 1
    path.append(init)

    # draw the map
    m = folium.Map(
        location=[48, 4],
        zoom_start=1)

    for i,l in enumerate(loc):
        folium.Marker(
            [l[0], l[1]],
            popup=cities[i]
            ).add_to(m)
    points = [[loc[p][0], loc[p][1]] for p in path]
    folium.PolyLine(points, color="red", weight=2.5, opacity=0.8).add_to(m)

    m.save("./tsp.html")
