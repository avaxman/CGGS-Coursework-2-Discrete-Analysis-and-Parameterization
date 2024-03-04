import numpy as np
import scipy.linalg
from scipy.sparse import csc_matrix, coo_matrix, linalg, bmat, diags
from scipy.sparse.linalg import spsolve


def accumarray(indices, values):
    output = np.zeros((np.max(indices) + 1), dtype=values.dtype)
    indFlat = indices.flatten()
    valFlat = values.flatten()
    # for index in range(indFlat.shape[0]):
    #     output[indFlat[index]] += valFlat[index]
    np.add.at(output, indFlat, valFlat)

    return output


def load_off_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the vertices and faces from the OFF file
    num_vertices, num_faces, _ = map(int, lines[1].split())

    vertices = np.array([list(map(float, line.split())) for line in lines[2:2 + num_vertices]])
    faces = np.array([list(map(int, line.split()))[1:] for line in lines[2 + num_vertices:]])

    return vertices, faces


def compute_areas_normals(vertices, faces):
    face_vertices = vertices[faces]

    # Compute vectors on the face
    vectors1 = face_vertices[:, 1, :] - face_vertices[:, 0, :]
    vectors2 = face_vertices[:, 2, :] - face_vertices[:, 0, :]

    # Compute face normals using cross product
    normals = np.cross(vectors1, vectors2)
    faceAreas = 0.5 * np.linalg.norm(normals, axis=1)

    normals /= (2.0 * faceAreas[:, np.newaxis])
    return normals, faceAreas


def visualize_boundary_Edges(ps_mesh, vertices, boundEdges):
    boundVertices = vertices[boundEdges]

    boundVertices = boundVertices.reshape(2 * boundVertices.shape[0], 3)
    curveNetIndices = np.arange(0, boundVertices.shape[0])
    curveNetIndices = curveNetIndices.reshape(int(len(curveNetIndices) / 2), 2)
    ps_net = ps.register_curve_network("boundary edges", boundVertices, curveNetIndices)

    return ps_net


def createEH(edges, halfedges):
    # Create dictionaries to map halfedges to their indices
    halfedges_dict = {(v1, v2): i for i, (v1, v2) in enumerate(halfedges)}
    # reversed_halfedges_dict = {(v2, v1): i for i, (v1, v2) in enumerate(halfedges)}

    EH = np.zeros((len(edges), 2), dtype=int)

    for i, (v1, v2) in enumerate(edges):
        # Check if the halfedge exists in the original order
        if (v1, v2) in halfedges_dict:
            EH[i, 0] = halfedges_dict[(v1, v2)]
        # Check if the halfedge exists in the reversed order
        if (v2, v1) in halfedges_dict:
            EH[i, 1] = halfedges_dict[(v2, v1)]

    return EH


def compute_edge_list(vertices, faces, sortBoundary=False):
    halfedges = np.empty((3 * faces.shape[0], 2))
    for face in range(faces.shape[0]):
        for j in range(3):
            halfedges[3 * face + j, :] = [faces[face, j], faces[face, (j + 1) % 3]]

    edges, firstOccurence, numOccurences = np.unique(np.sort(halfedges, axis=1), axis=0, return_index=True,
                                                     return_counts=True)
    edges = halfedges[np.sort(firstOccurence)]
    edges = edges.astype(int)
    halfedgeBoundaryMask = np.zeros(halfedges.shape[0])
    halfedgeBoundaryMask[firstOccurence] = 2 - numOccurences
    edgeBoundMask = halfedgeBoundaryMask[np.sort(firstOccurence)]

    boundEdges = edges[edgeBoundMask == 1, :]
    boundVertices = np.unique(boundEdges).flatten()

    # EH = [np.where(np.sort(halfedges, axis=1) == edge)[0] for edge in edges]
    # EF = []

    EH = createEH(edges, halfedges)
    EF = np.column_stack((EH[:, 0] // 3, (EH[:, 0] + 2) % 3, EH[:, 1] // 3, (EH[:, 1] + 2) % 3))

    if (sortBoundary):
        loop_order = []
        loopEdges = boundEdges.tolist()
        current_node = boundVertices[0]  # Start from any node
        visited = set()
        while True:
            loop_order.append(current_node)
            visited.add(current_node)
            next_nodes = [node for edge in loopEdges for node in edge if
                          current_node in edge and node != current_node and node not in visited]
            if not next_nodes:
                break
            next_node = next_nodes[0]
            loopEdges = [edge for edge in loopEdges if
                          edge != (current_node, next_node) and edge != (next_node, current_node)]
            current_node = next_node
            current_node = next_node

        boundVertices = np.array(loop_order)

    return halfedges, edges, edgeBoundMask, boundVertices, EH, EF


def compute_angle_defect(vertices, faces, boundVertices):
    #TODO: complete
    return np.zeros((vertices.shape[0])) #stub


def compute_mean_curvature_normal(vertices, faces, faceNormals, L, vorAreas):
    #TODO: complete
    #stubs
    MCNormal = np.zeros((vertices.shape[0],3)) #stub
    MC = np.zeros(vertices.shape[0])
    vertexNormals = np.zeros((vertices.shape[0],3))
    return MCNormal, MC, vertexNormals


def compute_laplacian(vertices, faces, edges, edgeBoundMask, EF):
    #TODO: complete

    #stubs
    L = []
    vorAreas = np.ones((vertices.shape[0]))
    d0 = []
    W = []
    return L, vorAreas, d0, W


def mean_curvature_flow(faces, boundVertices, currVertices, L, vorAreas, flowRate, isExplicit):
    #TODO: complete
    return currVertices #stub


def compute_boundary_embedding(vertices, boundVertices, r):
    #TODO: complete
    return np.zeros((boundVertices.shape[0],2)) #stub


def compute_tutte_embedding(vertices, faces, d0, W, boundVertices, boundUV):
    #TODO: complete
    return np.zeros((vertices.shape[0],2)) #stub
