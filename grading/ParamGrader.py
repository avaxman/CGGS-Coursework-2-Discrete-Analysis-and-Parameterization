import os
import sys
import pickle
import numpy as np

sys.path.append(os.path.join('..', 'code'))
from DAFunctions import load_off_file, compute_edge_list, compute_laplacian, compute_areas_normals, compute_boundary_embedding, compute_tutte_embedding

if __name__ == '__main__':

    data_path = os.path.join('..', 'data', 'param')  # Replace with the path to your folder

    # Get a list of all files in the folder with the ".off" extension
    off_files = [file for file in os.listdir(data_path) if file.endswith(".off")]

    for currFileIndex in range(len(off_files)):
        print("Processing mesh ", off_files[currFileIndex])
        off_file_path = os.path.join(data_path, off_files[currFileIndex])
        vertices, faces = load_off_file(off_file_path)

        root, old_extension = os.path.splitext(off_file_path)
        pickle_file_path = root + '-param.data'
        with open(pickle_file_path, 'rb') as pickle_file:
            loaded_data = pickle.load(pickle_file)

        halfedges, edges, edgeBoundMask, boundVertices, EH, EF = compute_edge_list(vertices, faces,
                                                                                   sortBoundary=True)
        L, vorAreas, d0, W = compute_laplacian(vertices, faces, edges, edgeBoundMask, EF)
        r = 5.5
        boundUV = compute_boundary_embedding(vertices, boundVertices, r)
        UV = compute_tutte_embedding(vertices, faces, d0, W, boundVertices, boundUV)

        print("boundUV error: ", np.max(np.abs(loaded_data['boundUV'] - boundUV)))
        print("UV error: ", np.max(np.abs(loaded_data['UV'] - UV)))
