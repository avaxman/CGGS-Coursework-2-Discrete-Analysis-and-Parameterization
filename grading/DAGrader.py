import os
import sys
import pickle
import numpy as np

sys.path.append(os.path.join('..', 'code'))
from DAFunctions import load_off_file, compute_edge_list, compute_laplacian, compute_areas_normals, \
    compute_mean_curvature_normal, compute_angle_defect

if __name__ == '__main__':

    data_path = os.path.join('..', 'data')  # Replace with the path to your folder

    # Get a list of all files in the folder with the ".off" extension
    off_files = [file for file in os.listdir(data_path) if file.endswith(".off")]

    for currFileIndex in range(len(off_files)):
        print("Processing mesh ", off_files[currFileIndex])
        off_file_path = os.path.join(data_path, off_files[currFileIndex])

        root, old_extension = os.path.splitext(off_file_path)
        pickle_file_path = root + '-discrete-analysis.data'
        with open(pickle_file_path, 'rb') as pickle_file:
            loaded_data = pickle.load(pickle_file)

        vertices, faces = load_off_file(off_file_path)
        halfedges, edges, edgeBoundMask, boundVertices, EH, EF = compute_edge_list(vertices, faces)

        L, vorAreas, d0, W = compute_laplacian(vertices, faces, edges, edgeBoundMask, EF)
        faceNormals, faceAreas = compute_areas_normals(vertices, faces)
        MCNormals, MC, vertexNormals = compute_mean_curvature_normal(vertices, faces, faceNormals, L, vorAreas)
        angleDefect = compute_angle_defect(vertices, faces, boundVertices)

        print("L error: ", np.max(np.abs(loaded_data['L'] - L)))
        print("vorAreas error: ", np.max(np.abs(loaded_data['vorAreas'] - vorAreas)))
        print("d0 error: ", np.max(np.abs(loaded_data['d0'] - d0)))
        print("W error: ", np.max(np.abs(loaded_data['W'] - W)))
        print("MCNormals error: ", np.max(np.abs(loaded_data['MCNormals'] - MCNormals)))
        print("MC error: ", np.max(np.abs(loaded_data['MC'] - MC)))
        print("vertexNormals error: ", np.max(np.abs(loaded_data['vertexNormals'] - vertexNormals)))
        print("angleDefect error: ", np.max(np.abs(loaded_data['angleDefect'] - angleDefect)))
