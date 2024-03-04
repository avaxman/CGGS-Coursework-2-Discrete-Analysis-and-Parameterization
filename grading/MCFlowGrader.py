import os
import sys
import pickle
import numpy as np

sys.path.append(os.path.join('..', 'code'))
from DAFunctions import load_off_file, compute_edge_list, compute_laplacian, compute_areas_normals, mean_curvature_flow

if __name__ == '__main__':

    data_path = os.path.join('..', 'data')  # Replace with the path to your folder

    # Get a list of all files in the folder with the ".off" extension
    off_files = [file for file in os.listdir(data_path) if file.endswith(".off")]
    for currFileIndex in range(len(off_files)):
        print("Processing mesh ", off_files[currFileIndex])
        off_file_path = os.path.join(data_path, off_files[currFileIndex])

        root, old_extension = os.path.splitext(off_file_path)
        pickle_file_path = root + '-MC-flow.data'
        with open(pickle_file_path, 'rb') as pickle_file:
            loaded_data = pickle.load(pickle_file)

        vertices, faces = load_off_file(off_file_path)
        halfedges, edges, edgeBoundMask, boundVertices, EH, EF = compute_edge_list(vertices, faces)
        L, vorAreas, d0, W = compute_laplacian(vertices, faces, edges, edgeBoundMask, EF)
        faceNormals, faceAreas = compute_areas_normals(vertices, faces)

        flowRate = 0.1 * np.min(vorAreas)
        currVertices1 = mean_curvature_flow(faces, boundVertices, vertices, L, vorAreas, flowRate, isExplicit=True)
        flowRate = 0.5 * np.min(vorAreas)
        currVertices2 = mean_curvature_flow(faces, boundVertices, currVertices1, L, vorAreas, flowRate,
                                            isExplicit=False)
        flowRate = 10.0 * np.min(vorAreas)
        currVertices3 = mean_curvature_flow(faces, boundVertices, currVertices2, L, vorAreas, flowRate,
                                            isExplicit=False)

        print("currVertices1 error: ", np.max(np.abs(loaded_data['currVertices1'] - currVertices1)))
        print("currVertices2 error: ", np.max(np.abs(loaded_data['currVertices2'] - currVertices2)))
        print("currVertices3 error: ", np.max(np.abs(loaded_data['currVertices3'] - currVertices3)))
