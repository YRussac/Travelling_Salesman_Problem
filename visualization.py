import matplotlib.pyplot as plt
import numpy as np
import folium

a = np.random.random((16, 16))
a = np.matrix(a)

def heatmap(m, labels, title=''):
    plt.imshow(m, cmap='RdYlGn')
    plt.xticks(list(range(len(labels))), labels, rotation=45)
    plt.yticks(list(range(len(labels))), labels)
    plt.title(title)
    plt.colorbar()
    plt.show()


def create_map(path):
    m = folium.Map(
        location=[45.5236, -122.6750],
        zoom_start=8)
    folium.RegularPolygonMarker(
        [45.5236, -122.6750],
        fill_color='#43d9de',
        radius=10,
        popup='test'
        ).add_to(m)
    folium.RegularPolygonMarker(
        [46.5236, -121.6750],
        fill_color='#43d9de',
        radius=10,
        popup='test2'
        ).add_to(m)
    folium.PolyLine([[45.5236, -122.6750], [46.5236, -121.6750]], color="red", weight=2.5, opacity=0.8).add_to(m)
